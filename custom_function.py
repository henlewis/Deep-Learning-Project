import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm

import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from PIL import Image as PilImage
from ultralytics import YOLO

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


# ===== IMAGE EXPLORATION =====
def explore_image_files(file_paths, explore_values=False):

    # Open each image to extract extra info
    image_sizes = []
    color_channels = []
    formats = []
    min_vals = []
    max_vals = []
    ratios = []

    # Iteration loop for each folder to compare the image sizes
    for file_path in file_paths:
        with PilImage.open(file_path) as img:
            image_sizes.append(img.size)
            color_channels.append(img.mode)
            formats.append(img.format)
            ratios.append(img.size[0]/img.size[1])

            if explore_values:
                # Convert to numpy to check the actual data type (Takes alot of time!!!)
                img_array = np.array(img)

                # Value range
                min_vals.append(img_array.min())
                max_vals.append(img_array.max())

    if explore_values:
        return image_sizes, color_channels, formats, ratios, min_vals, max_vals
    else:
        return image_sizes, color_channels, formats, ratios
    

def smallest_largest_image(metadata, directory):
    
    # Find largest and smallest images by pixel count
    largest = metadata.loc[metadata["pixel_count"].idxmax()]
    smallest = metadata.loc[metadata["pixel_count"].idxmin()]

    largest_image_path = directory + largest["file_path"]
    largest_pixels = largest["pixel_count"]
    largest_dims = (largest["width"], largest["height"])

    smallest_image_path = directory + smallest["file_path"]
    smallest_pixels = smallest["pixel_count"]
    smallest_dims = (smallest["width"], smallest["height"])

    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Largest image
    img_largest = PilImage.open(largest_image_path)
    axes[0].imshow(img_largest)
    axes[0].axis('off')
    axes[0].set_title(
        f"LARGEST IMAGE\n{os.path.basename(largest_image_path)}\n"
        f"{largest_dims[0]} x {largest_dims[1]} pixels\n"
        f"({largest_pixels:,} total pixels)",
        fontsize=12, fontweight='bold'
    )

    # Smallest image
    img_smallest = PilImage.open(smallest_image_path)
    axes[1].imshow(img_smallest)
    axes[1].axis('off')
    axes[1].set_title(
        f"SMALLEST IMAGE\n{os.path.basename(smallest_image_path)}\n"
        f"{smallest_dims[0]} x {smallest_dims[1]} pixels\n"
        f"({smallest_pixels:,} total pixels)",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.show()

    
    print(f"Largest image path:  {largest_image_path}")
    print(f"\nSmallest image path: {smallest_image_path}")



# ===== DATA PREPROCESSING =====
def get_distances(metadata, centroids):

    # Calculate distances to centroids
    distances = []

    for idx, row in metadata.iterrows():
        family = row['family']
        if family in centroids:
            # Calculate Euclidean distance
            distance = np.linalg.norm(row['embedding'] - centroids[family])
            distances.append(distance)
        else:
            distances.append(np.nan)

    return distances


def distance_to_centroid(train_df, test_df):

    # Calculate a centroid for each family based on the embedings from the train_df
    centroids = {}

    for family in train_df['family'].unique():
        # Get all the embedings for 1 family
        family_embeddings = train_df[train_df['family'] == family]['embedding'].tolist()

        if len(family_embeddings) > 0:
            # Stack embeddings and calculate mean
            embeddings_array = np.stack(family_embeddings)
            centroids[family] = np.mean(embeddings_array, axis=0)
    
    # Return the distances for both train and test based on the centroids of the train
    return get_distances(train_df, centroids), get_distances(test_df, centroids)


def find_people_yolo(metadata, directory, model_id, conf_threshold, person_class_id):
    # Find images containing people using YOLO

    model = YOLO(model_id)
    
    has_person = []
    person_confidence = []
    person_bbox = []
    
    print(f"Scanning for people with confidence > {conf_threshold}.")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="YOLO person detection"):
        filepath = os.path.join(directory, row['file_path'])
        
        try:
            results = model.predict(
                filepath,
                verbose=False,
                conf=conf_threshold,
                iou=0.5,
                classes=[person_class_id]
            )
            
            boxes = results[0].boxes if results and results[0].boxes is not None else None
            
            if boxes and len(boxes) > 0:
                has_person.append(True)
                person_confidence.append(float(boxes.conf.max()))

                # Save box around the object as [x1, y1, x2, y2] for visualization
                best_box_idx = boxes.conf.argmax()
                person_bbox.append(boxes.xyxy[best_box_idx].tolist())
            else:
                has_person.append(False)
                person_confidence.append(0.0)
                person_bbox.append(None)
                
        except Exception as e:
            has_person.append(False)
            person_confidence.append(0.0)
    
    return has_person, person_confidence, person_bbox


def compute_clip_scores_batch(metadata, directory, clip_processor, clip_model, text_prompts, text_features, batch_size=32):
    # Process all images and compute CLIP semantic scores

    all_results = []
    file_paths = metadata['file_path'].tolist()
    
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Computing CLIP scores"):
        batch_paths = file_paths[i:i+batch_size]
        batch_images = []
        batch_indices = []
        
        for j, path in enumerate(batch_paths):
            try:
                full_path = directory + path
                img = PilImage.open(full_path).convert("RGB")
                batch_images.append(img)
                batch_indices.append(i + j)
            except Exception as e:
                all_results.append({
                    'clip_photo_score': None,
                    'clip_noise_score': None,
                    'clip_semantic_quality': None,
                    'clip_best_match': None
                })
        
        if not batch_images:
            continue
        
        image_inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).cpu().numpy()
        
        for sims in similarities:
            photo_score = sims[:3].max()
            noise_score = sims[3:].max()
            all_results.append({
                'clip_photo_score': float(photo_score),
                'clip_noise_score': float(noise_score),
                'clip_semantic_quality': float(photo_score - noise_score),
                'clip_best_match': text_prompts[sims.argmax()]
            })
    
    return pd.DataFrame(all_results)


def check_animal_presence_batch(img_path, model, threshold):
    # Check if an image contains an animal using ImageNet predictions
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Get predictions
    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=10)[0]
    
    # Get top prediction
    top_class = decoded[0][1]
    top_prob = float(decoded[0][2])
    
    # Check if any animal class has high probability
    animal_prob = 0.0
    is_animal = False
    
    for pred in decoded:
        class_idx = pred[0]
        class_name = pred[1]
        prob = float(pred[2])
        
        # Check classes by keywords
        animal_keywords = [
            'dog', 'cat', 'bird', 'fish', 'snake', 'turtle', 'frog',
            'insect', 'butterfly', 'spider', 'crab', 'lobster', 'snail',
            'worm', 'jellyfish', 'coral', 'lion', 'tiger', 'bear',
            'elephant', 'monkey', 'ape', 'whale', 'dolphin', 'seal',
            'penguin', 'eagle', 'owl', 'parrot', 'lizard', 'crocodile',
            'salamander', 'scorpion', 'beetle', 'bee', 'ant', 'fly',
            'cockroach', 'mantis', 'dragonfly', 'moth', 'slug'
        ]
        
        # Check if it's likely an animal
        if any(keyword in class_name.lower() for keyword in animal_keywords):
            if prob > animal_prob:
                animal_prob = prob
                is_animal = animal_prob >= threshold
    
    return is_animal, animal_prob, top_class, top_prob


def check_animal_presence(metadata, directory, imagenet_model, threshold=0.02):
 
    is_animal_list = []
    animal_prob_list = []
    top_class_list = []
    top_prob_list = []

    # Run the batched function to collect all the results
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="ImageNet animal detection"):
        img_path = os.path.join(directory, row['file_path'])
        is_animal, animal_prob, top_class, top_prob = check_animal_presence_batch(
            img_path, imagenet_model, threshold=threshold
        )
        is_animal_list.append(is_animal)
        animal_prob_list.append(animal_prob)
        top_class_list.append(top_class)
        top_prob_list.append(top_prob)

    return is_animal_list, animal_prob_list, top_class_list, top_prob_list


def generate_embeddings(metadata, directory, image_size, embedding_model):

    # Create generator for the model
    datagen = ImageDataGenerator() 

    # Create generator 
    preprocessing_generator = datagen.flow_from_dataframe(
        dataframe=metadata,
        directory=directory,
        x_col='file_path',
        y_col='family',  
        target_size=image_size,
        batch_size=32,  
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False       # Not to mess up the order of data
    )

    embeddings_list = []

    # Calculate number of batches
    num_batches = len(preprocessing_generator)

    for i in tqdm(range(num_batches), desc="Embedding images"):     # Add a fancy progress bar
        batch_images, _ = next(preprocessing_generator)             # Get images (ignore labels)
        
        # Generate embeddings
        batch_embeddings = embedding_model.predict(batch_images, verbose=0) 
        embeddings_list.extend(batch_embeddings)

    return embeddings_list



# ===== VISUALIZATIONS =====
# -----      PLOTS     -----
def plot_phylum_counts(metadata):
    # Get phylum counts
    phylum_counts = metadata['phylum'].value_counts()

    # Calculate number of families per phylum
    families_per_phylum = metadata.groupby('phylum')['family'].nunique()

    # Create custom hover data with family counts
    hover_data = []
    for phylum in phylum_counts.index:
        image_count = phylum_counts[phylum]
        family_count = families_per_phylum[phylum]
        percentage = (image_count / len(metadata)) * 100
        hover_data.append([image_count, percentage, family_count])

    # Convert to numpy array
    hover_data = np.array(hover_data)

    # Create figure
    fig = go.Figure(go.Bar(
        x=phylum_counts.index,
        y=phylum_counts.values,
        marker=dict(
            color='#6366f1',
            line=dict(color='#4f46e5', width=0.5)
        ),
        text=phylum_counts.values,
        textposition='outside',
        textfont=dict(size=12),
        customdata=hover_data,
        hovertemplate='<b>%{x}</b><br>' +
                    'Images: <b>%{y}</b> (<b>%{customdata[1]:.2f}%</b>)<br>' +
                    'Families: %{customdata[2]}<br><extra></extra>'
        
    ))

    fig.update_layout(
        title={
            'text': '<b>Distribution of Species by Phylum</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Phylum',
        yaxis_title='Count',
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(size=11),
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        font=dict(size=11),
        showlegend=False
    )

    fig.show()


def plot_family_counts(metadata):
    # Get all family counts
    family_counts = metadata['family'].value_counts()

    # Create figure with scrollable y-axis
    fig = go.Figure(go.Bar(
        x=family_counts.values,
        y=family_counts.index,
        orientation='h',
        marker=dict(
            color='#6366f1',
            line=dict(color='#4f46e5', width=0.5)
        ),
        text=family_counts.values,
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>Images: <b>%{x}</b> (<b>%{customdata:.2f}%</b>)<extra></extra>',
        customdata=(family_counts.values / len(metadata)) * 100
    ))

    fig.update_layout(
        title={
            'text': '<b>Distribution Of Species By Family</b><br>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Number of Images',
        yaxis_title='Family',
        height=max(1000, len(family_counts) * 15),  # Dynamic height based on number of families
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            autorange='reversed',  # Highest count on top
            tickfont=dict(size=9),
            showgrid=False
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        font=dict(size=11),
        margin=dict(l=200, r=100, t=100, b=50), 
        showlegend=False
    )

    fig.show()


def plot_taxonomic_hierarchy(metadata):

    phylum_to_family = metadata.groupby(['phylum', 'family']).size().reset_index(name='count')

    # Prepare labels
    phylum_labels = list(metadata['phylum'].unique())
    family_labels = list(metadata['family'].unique())
    labels = phylum_labels + family_labels
    label_dict = {label: idx for idx, label in enumerate(labels)}

    labels_display = []
    for label in labels:
        if label in phylum_labels:
            labels_display.append(label)
        else:
            labels_display.append('')     # Hide family labels


    # Prepare sources,targets and values
    source = []
    target = []
    value = []
    link_hover = []

    for _, row in phylum_to_family.iterrows():
        source_idx = label_dict[row['phylum']]
        target_idx = label_dict[row['family']]
        count = row['count']
        
        source.append(source_idx)
        target.append(target_idx)
        value.append(count)
        
        # Custom hover text for links
        link_hover.append(f"Phylum: <b>{row['phylum']}</b><br>Family: <b>{row['family']}</b><br>Images: <b>{count}</b>")

    # Custom hover for nodes
    node_hover = []
    for label in labels:
        if label in phylum_labels:
            total = metadata[metadata['phylum'] == label].shape[0]
            node_hover.append(f"Phylum: <b>{label}</b><br>Total Images: <b>{total}</b>")
        else:
            total = metadata[metadata['family'] == label].shape[0]
            node_hover.append(f"Family: <b>{label}</b><br>Total Images: <b>{total}</b>")

    # Plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels_display,
            color='#6366f1',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=node_hover
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(99, 102, 241, 0.3)',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=link_hover
        )
    )])

    fig.update_layout(
        title='<b>Taxonomic Hierarchy: Phylum → Family</b>',
        font=dict(size=12),
        height=1200,
        width=1200
    )

    fig.show()


def plot_color_chanels_counts(metadata):
    
    # Plot colour channel distribution
    color_channel_mapping = {
        'L': 'Greyscale',
        'RGB': 'RGB',
        'RGBA': 'RGBA',
        'P': 'Palette',
        'CMYK': 'CMYK',
        '1': 'Binary',
        'LA': 'Greyscale + Alpha'
    }

    # Get value counts and map to readable names
    color_counts = metadata['color_channel'].value_counts()
    color_counts.index = color_counts.index.map(lambda x: color_channel_mapping.get(x, x))

    # Create figure
    fig = go.Figure(go.Bar(
        x=color_counts.index,
        y=color_counts.values,
        marker=dict(
            color='#6366f1',
            line=dict(color='#4f46e5', width=0.5)
        ),
        text=color_counts.values,
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate='<b>%{x}</b><br>Images: <b>%{y}</b><br>Percentage: <b>%{customdata:.2f}%</b><extra></extra>',
        customdata=(color_counts.values / len(metadata)) * 100
    ))

    fig.update_layout(
        title={
            'text': '<b>Distribution Of Color Channels</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Color Channel',
        yaxis_title='Count',
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(size=11),
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        font=dict(size=11),
        showlegend=False
    )

    fig.show()


def plot_image_size_scatter(metadata, common_ratio, width_max):

    fig = go.Figure()

    # Create scatter plot of image sizes
    fig.add_trace(go.Scatter(
        x=metadata['width'],
        y=metadata['height'],
        mode='markers',
        marker=dict(
            size=4,
            color='#6366f1',
            opacity=0.4,
            line=dict(width=0)
        ),
        name='Images',
        hovertemplate='Width: <b>%{x}px</b><br>Height:<b> %{y}px</b><extra></extra>'
    ))

    # Add aspect ratio reference line
    max_y = metadata['height'].max()
    max_x_for_ratio = common_ratio * max_y
    fig.add_trace(go.Scatter(
        x=[0, max_x_for_ratio],
        y=[0, max_y],
        mode='lines',
        line=dict(color='#ef4444', width=2.5, dash='dash'),
        name=f'{round(common_ratio*3)}:3 Aspect Ratio',
        hoverinfo='skip'
    ))

    # Add width threshold line
    fig.add_trace(go.Scatter(
        x=[width_max, width_max],
        y=[0, max_y],
        mode='lines',
        line=dict(color='#10b981', width=2.5, dash='dash'),
        name=f'Max Width ({width_max}px)',
        hoverinfo='skip'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Image Size Distribution Analysis</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#1f2937'}
        },
        xaxis=dict(
            title='Width (pixels)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, color='#374151')
        ),
        yaxis=dict(
            title='Height (pixels)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, color='#374151')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#d1d5db',
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode='closest'
    )

    fig.show()


def plot_yolo_detections(metadata, directory, N, n_cols=10):
    # Display images where YOLO detected people

    person_images = metadata[metadata['has_person'] == True].copy()
    person_images = person_images.sort_values('person_confidence', ascending=False)
    
    if len(person_images) == 0:
        print("No images with people detected by YOLO!")
        return
    
    # Take a sample for plotting
    person_images = person_images.sample(min(N, len(person_images)), random_state=42)

    print(f"Showing {len(person_images)} images with people detected by YOLO\n")
    
    n_rows = math.ceil(len(person_images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2.5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (_, row) in enumerate(person_images.iterrows()):
        try:
            img = PilImage.open(directory + row['file_path'])
            axes[idx].imshow(img)

            # Add a bounding box
            if row['person_bbox'] is not None:
                x1, y1, x2, y2 = row['person_bbox']
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor='limegreen', facecolor='none')
                axes[idx].add_patch(rect)


            axes[idx].set_title(f"conf: {row['person_confidence']:.2f}", fontsize=6)
        except:
            pass
        axes[idx].axis('off')
    
    for idx in range(len(person_images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"YOLO Person Detections ({len(person_images)} images)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_non_animal_images(metadata, directory, n_rows=5, n_cols=10):
    # Display images where ImageNet didn't detect animals

    non_animals = metadata[metadata['has_animal'] == False].copy()
    non_animals = non_animals.sort_values('animal_confidence', ascending=True)
    
    if len(non_animals) == 0:
        print("All images contain animals!")
        return
    
    n_show = min(n_cols * n_rows, len(non_animals))
    print(f"Showing {n_show} of {len(non_animals)} non-animal images detected by InceptionV3\n")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2.5*n_rows))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(non_animals.head(n_show).iterrows()):
        try:
            img = PilImage.open(directory + row['file_path'])
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"{row['predicted_class'][:12]}\n{row['animal_confidence']:.3f}",
                fontsize=6
            )
        except:
            pass
        axes[idx].axis('off')
    
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"ImageNet Non-Animal Images ({len(non_animals)} total)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_clip_outliers(metadata, directory, N, n_cols=10):
    # Display CLIP outliers

    bad_images = metadata[metadata['clip_semantic_quality'] < 0].copy()
    bad_images = bad_images.sort_values('clip_semantic_quality', ascending=True)
    
    if len(bad_images) == 0:
        print("No CLIP outliers found!")
        return
    
    print(f"CLIP Category breakdown (total {len(bad_images)}):")
    print(bad_images['clip_best_match'].value_counts())
    print("\n" + "="*60 + "\n")

    # Take a sample for plotting
    bad_images = bad_images.sample(min(N, len(bad_images)))
    
    print(f"Showing {len(bad_images)} CLIP outliers\n")
    
    n_rows = math.ceil(len(bad_images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2.5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (_, row) in enumerate(bad_images.iterrows()):
        try:
            img = PilImage.open(directory + row['file_path'])
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"{row['clip_best_match'].replace('a ', '').replace('an ', '')[:15]}\n"
                f"{row['clip_semantic_quality']:.2f}",
                fontsize=6
            )
        except:
            pass
        axes[idx].axis('off')
    
    for idx in range(len(bad_images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"CLIP Outliers ({len(bad_images)} images)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_model_results(history_main, metric, history_ft=None):

    # If there is extra fine tuning history
    if history_ft:
        # Extract metrics
        train_main = history_main.history[metric]
        val_main = history_main.history[f'val_{metric}']
        train_ft = history_ft.history[metric]
        val_ft = history_ft.history[f'val_{metric}']

        # Combine
        train_all = train_main + train_ft
        val_all = val_main + val_ft

        # Epoch where FT starts
        ft_start = len(train_main)

        # Add a note to title
        title = '(Main + Fine-Tune)'
    
    else:
        # If there is no fine tuning history
        train_all = history_main.history[metric]
        val_all = history_main.history[f'val_{metric}']
        title = ''

    # Epochs total
    epochs = np.arange(1, len(train_all) + 1)

    # Build figure
    fig = go.Figure()

    # Train line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_all,
        mode='lines',
        name='Train',
        line=dict(width=2, color='#6366f1'),
        hovertemplate='Train ' + metric.title() + ': <b>%{y:.4f}</b><extra></extra>'
    ))

    # Validation line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_all,
        mode='lines',
        name='Validation',
        line=dict(width=2, color='#ffa500'),
        hovertemplate='Validation ' + metric.title() + ': <b>%{y:.4f}</b><extra></extra>'
    ))

    # If there is extra fine tuning history
    if history_ft:
        # Add a vertical line at fine-tuning start
        fig.add_vline(
            x=ft_start,
            line_width=2,
            line_dash="dash",
            line_color="tomato",
            annotation_text="Fine-tuning start",
            annotation_position="top"
        )

    # Layout
    fig.update_layout(
        hovermode='x unified',
        title={
            'text': f'<b>Model {metric.title()} {title}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Epochs',
        yaxis_title=metric.title(),
        height=600,
        width=1000,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=11),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.05,
            font=dict(size=12)
        )
    )

    fig.show()


def plot_confusion_matrix(cm):

    # Normalize for better color scaling
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(cm_normalized, 
                annot=False, 
                cmap='Blues',  
                square=True,
                linewidths=0,
                ax=ax)

    # Add title and labels
    plt.title('Confusion Matrix: Predictions vs Actual Labels', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

    # Add grid for better readability
    for i in range(0, 200, 10):
        ax.axhline(y=i, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(x=i, color='white', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----     IMAGES     -----
def plot_images_from_directory(directory, img_per_class=5):

    # Print example images of each class
    for label in os.listdir(directory):
        path = directory + str(label)

        if not os.path.isdir(path):
            print(f"Directory {path} does not exist.")
            continue

        folder_data = os.listdir(path)
        k = 0
        print(f'{label} ({len(folder_data)} images)')

        # Collect image paths
        image_paths = []
        for image_path in folder_data:
            if k < img_per_class:                                               
                full_path = os.path.join(path, image_path)
                image_paths.append(full_path)
                k += 1

        # Display images
        if image_paths:
            fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 3))
            if len(image_paths) == 1:
                axes = [axes]

            for ax, img_path in zip(axes, image_paths):
                img = PilImage.open(img_path)
                ax.imshow(img)
                ax.axis('off')

            plt.tight_layout()
            plt.show()


def plot_images_from_generator(generator, num_batches=False):

    # Get class names from the generator
    class_names = list(generator.class_indices.keys())

    if not num_batches:
        # If number of batches not set, plot all
        num_batches = int(math.ceil(generator.n / generator.batch_size))

    for i in range(num_batches):
        images, labels = next(generator)

        # Plot the images in the current batch
        batch_size_actual = images.shape[0]
        n_cols = min(8, batch_size_actual)                  # <-- max columns per row
        n_rows = math.ceil(batch_size_actual / n_cols)

        plt.figure(figsize=(3 * n_cols, 3 * n_rows))

        for j in range(batch_size_actual):
            ax = plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(images[j])

            # Get the class name
            label_idx = np.argmax(labels[j])
            plt.title(class_names[label_idx], fontsize=9)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def plot_furthest_images_batched(directory, metadata, N, n_cols=10):
    # Get the images sorted by distance to centroid (make a copy to not mess up the original metadata)
    sorted_metadata = metadata.sort_values(by='distance_to_centroid', ascending=False).copy()

    for i in range(0, N, n_cols):
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 2))

        # Flatten axes to simplify indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx in range(i, min(i + n_cols, N)):
            if idx < len(sorted_metadata):
                # Show the image with family and ID in the title
                img_path = os.path.join(directory, sorted_metadata.iloc[idx]['file_path'])
                img = PilImage.open(img_path)
                axes[idx - i].imshow(img)
                family = sorted_metadata.iloc[idx]['family']
                axes[idx - i].set_title(f'{family} (id={idx})', fontsize=8)
                axes[idx - i].axis('off')

            else:
                # Hide the subplot if there is no image
                axes[idx - i].axis('off')

        plt.tight_layout()
        plt.show()


def plot_good_vs_bad_images(good_images_df, outliers, directory, good_images=2):

    for family in outliers['family'].unique():
        # Get all the image paths sorted
        image_paths = list(directory+good_images_df[good_images_df.family == family].sort_values('distance_to_centroid')['file_path'])[:good_images]
        image_paths.extend(list(directory+outliers[outliers.family == family].sort_values('distance_to_centroid', ascending=False)['file_path']))

        if image_paths:
            # Set number of columns and calculate rows needed
            n_cols = 7
            n_images = len(image_paths)
            n_rows = (n_images + n_cols - 1) // n_cols 

            # Create subplots with appropriate dimensions
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows))

            # Flatten axes array for easier iteration
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # Display images
            for idx, img_path in enumerate(image_paths):
                img = PilImage.open(img_path).resize((224, 224), PilImage.LANCZOS)
                axes[idx].imshow(img)
                axes[idx].axis('off')

            # Add green outline for good images
                if idx < good_images:
                    rect = plt.Rectangle((0, 0), 223, 223, fill=False,
                                        edgecolor='lime', linewidth=3)
                    axes[idx].add_patch(rect)

            # Hide any unused subplots
            for idx in range(n_images, n_rows * n_cols):
                axes[idx].axis('off')

            plt.suptitle(f"{family}, {good_images_df[good_images_df.family == family].shape[0]} good images left")
            plt.tight_layout()
            plt.show()


def plot_datagen_sample(generator): 
    
   # Get a batch of images
    images, labels = next(generator)

    # Get class names
    class_names = list(generator.class_indices.keys())

    # Plot the images
    plt.figure(figsize=(12, 8))

    for i in range(min(21, len(images))):
        ax = plt.subplot(3, 7, i + 1)
        # Denormalize for visualization
        img_display = images[i].copy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        plt.imshow(img_display)
        plt.title(class_names[int(labels[i])], fontsize=8)
        plt.axis("off")

    plt.suptitle("Sample Training Batch (with augmentation)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_misclassifications(test_generator, y_pred_classes, y_true, class_names, train_df, directory, N=5):
    
    # Find misclassified indices
    misclassified_idx = np.where(y_pred_classes != y_true)[0]    

    for i, idx in enumerate(misclassified_idx[:min(N, len(misclassified_idx))]):
        
        actual_calss = class_names[y_true[idx]]
        predicted_class = class_names[y_pred_classes[idx]]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # Plot the misclassified image
        img_path = test_generator.filepaths[idx] 
        img = plt.imread(img_path)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title("Misclassified Image")

        # Plot the image from the actual class
        img_path = directory + train_df[train_df.family == actual_calss].sort_values('distance_to_centroid')['file_path'].iloc[0]  
        img = plt.imread(img_path)
        axes[1].imshow(img)
        axes[1].axis('off')
        axes[1].set_title(f"Actual class:\n{actual_calss.title()}")

        # Add a green box around correct class
        height, width = img.shape[:2]
        rect = plt.Rectangle((0, 0), width, height, fill=False, edgecolor='lime', linewidth=5)
        axes[1].add_patch(rect)

        # Plot the image from the class that model predicted as
        img_path = directory + train_df[train_df.family == predicted_class].sort_values('distance_to_centroid')['file_path'].iloc[0]  
        img = plt.imread(img_path)
        axes[2].imshow(img)
        axes[2].axis('off')
        axes[2].set_title(f"Predicted class:\n{predicted_class.title()}")

         # Add a red box around incorrect class
        height, width = img.shape[:2]
        rect = plt.Rectangle((0, 0), width, height, fill=False, edgecolor='tomato', linewidth=5)
        axes[2].add_patch(rect)

        plt.tight_layout()
        plt.show()


# ===== CUSTOM PREDICTIONS =====
def predict_with_metadata(model, test_generator, test_df, phylum_to_families):
    all_preds = []

    # Reset the generator to the beginning
    test_generator.reset()

    # Loop over batches
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i] 
        batch_start = i * test_generator.batch_size
        batch_end = batch_start + x_batch.shape[0]
        
        # Get corresponding phylum from dataframe
        phylum_batch = test_df['phylum'].iloc[batch_start:batch_end].values
        
        # Get model logits
        logits = model.predict(x_batch)  # shape: (batch_size, 202)
        
        # Apply filtering per image
        batch_preds = []
        for j in range(len(x_batch)):
            phylum = phylum_batch[j]
            valid_families = phylum_to_families[phylum]

            mask = np.full_like(logits[j], -np.inf)
            mask[valid_families] = 0
            filtered_logits = logits[j] + mask

            probs = tf.nn.softmax(filtered_logits).numpy()
            pred_family = int(np.argmax(probs))
            batch_preds.append(pred_family)
        
        all_preds.extend(batch_preds)
    
    return np.array(all_preds)
