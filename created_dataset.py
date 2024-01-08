import json
import os
import random

import cv2


def find_annotations_for_image(image_name, annotations):
    for img in annotations['images']:
        if img['file_name'] == image_name:
            image_info = img
            break
    else:
        return None, None  # Image not found

    # Find all annotations for the given image
    image_annotations = [ann for ann in annotations['annotations']
                         if ann['image_id'] == image_info['id']]

    return image_info, image_annotations


def load_image(image_dir, image_info):
    image_path = f'{image_dir}/{image_info["file_name"]}'
    image = cv2.imread(image_path)
    return image


def crop_signature(image, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    return image[y:y+h, x:x+w]


def fill_check_field(bank_check, signature, target_bbox, pen_color,
                     random_position=True):
    x, y, w, h = target_bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    signature_height, signature_width = signature.shape[:2]

    # Calculate scaling factor (assuming you want to fit the signature within the target bbox)
    scaling_factor = min(w / signature_width, h / signature_height)
    new_width = int(signature_width * scaling_factor)
    new_height = int(signature_height * scaling_factor)

    # Resize the signature
    resized_signature = cv2.resize(signature, (new_width, new_height))

    # Create a mask for the signature (assuming white background)
    grayscale_signature = cv2.cvtColor(resized_signature, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grayscale_signature, 240,
                            255, cv2.THRESH_BINARY_INV)

    # Random position within the target bbox
    if random_position:
        max_x = x + w - new_width
        max_y = y + h - new_height

        # Ensure there's at least one valid position
        top_left_x = random.randint(x, max(max_x, x))
        top_left_y = random.randint(y, max(max_y, y))
    else:
        top_left_x = x
        top_left_y = y

    # Change the color of the signature
    # by replacing RGB values with the pen color
    colored_signature = resized_signature.copy()
    colored_signature[mask > 0] = pen_color

    # Create an alpha mask based on the mask to blend the signature
    alpha_mask = cv2.merge([mask, mask, mask]) / 255.0

    # Position the colored signature on the bank check
    for c in range(0, 3):
        region_of_interest = bank_check[top_left_y:top_left_y +
                                        new_height, top_left_x:top_left_x+new_width, c]
        colored_signature_pixel = colored_signature[:, :, c] * (
            alpha_mask[:, :, c])
        background_pixel = region_of_interest * (1 - alpha_mask[:, :, c])
        # Blend the signature and the background
        bank_check[top_left_y:top_left_y+new_height, top_left_x:top_left_x +
                   new_width, c] = colored_signature_pixel + background_pixel
        
    # Return new bounding box (in COCO format [x, y, width, height])
    new_bbox = [top_left_x, top_left_y, new_width, new_height]
    return bank_check, new_bbox


def load_dataset(dataset_dir):
    images_path = f'{dataset_dir}/data'
    annotations_path = f'{dataset_dir}/labels.json'
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    return images_path, annotations


def select_random_annotation(annotations, class_id):
    annotations = [ann for ann in annotations
                   if ann['category_id'] == class_id]
    return random.choice(annotations)


# Load the signature dataset
signatures_dir = 'data/forged_signatures'
signature_images_path, signature_annotations = load_dataset(signatures_dir)

# Load the checks dataset
checks_dir = 'data/checks'
checks_images_path, checks_annotations = load_dataset(checks_dir)

# Load the additional text dataset
texts_dir = 'data/additional_text'
texts_images_path, texts_annotations = load_dataset(texts_dir)

# Define the directory to save the composite images
output_dir = 'data/forged_checks'
output_images_path = f'{output_dir}/data'
os.makedirs(output_images_path, exist_ok=True)

pens = [
    ([0, 0, 0], "black"),
    ([0, 0, 0], "black"),
    ([64, 64, 64], "gray"),
    ([64, 64, 64], "gray"),
    ([139, 0, 0], "blue"),
    ([139, 0, 0], "blue"),
    ([139, 0, 0], "blue"),
    ([32, 32, 139], "red"),
    ([32, 32, 139], "red"),
    ([34, 139, 34], "green"),
]

forged_checks_data = {
    'info': signature_annotations['info'],
    'licenses': signature_annotations['licenses'],
    'categories': checks_annotations['categories'],
    'images': [],
    'annotations': [],
}
forged_checks_images = []
forged_checks_annotations = []

new_id = 1

signature_files = [img['file_name'] for img in signature_annotations['images']]
check_files = [img['file_name'] for img in checks_annotations['images']]

# Composite each signature on the bank check
for signature_file in signature_files:
    signature_id = signature_file.split('.')[0]

    # Get the annotations for the current signature
    image_info, image_annotations = find_annotations_for_image(
        signature_file, signature_annotations)

    # Load the signature image
    signatures_image = load_image(signature_images_path, image_info)

    for i, annotation in enumerate(image_annotations):
        cropped_signature = crop_signature(
            signatures_image, annotation['bbox'])
        if cropped_signature.shape[0] == 0 or cropped_signature.shape[1] == 0:
            print(f'Empty signature: {signature_file}, {annotation["bbox"]}')
            continue

        # Select 5 random checks to composite the signature on
        selected_checks = random.sample(check_files, 5)
        for bank_check_file in selected_checks:
            check_info, check_anno = find_annotations_for_image(
                bank_check_file, checks_annotations)
            target_bboxes = {ann['category_id']: ann['bbox']
                             for ann in check_anno}

            # Select a random pen color
            pen_color, pen_name = random.choice(pens)

            # Load the bank check image
            bank_check_id = bank_check_file.split('.')[0]
            filled_check = load_image(checks_images_path, check_info)

            # Composite the signature and other text on the check
            for cat_id, target_bbox in target_bboxes.items():
                # Each check has only one annotation, i.e. the target bbox
                target_bbox = target_bboxes[cat_id]

                # Fill the check with the signature
                if cat_id == 1: # bank check
                    filled_check, new_bbox = fill_check_field(filled_check, cropped_signature,
                                                                 target_bbox, pen_color)

                # Fill the check with the additional text
                else: # additional text
                    # Load a random text image
                    text_image = random.choice([img['file_name'] for img in texts_annotations['images']])
                    text_info, current_text_annotations = find_annotations_for_image(
                        text_image, texts_annotations)
                    text_image = load_image(texts_images_path, text_info)

                    # Select random text to add to the check
                    text_annotation = select_random_annotation(current_text_annotations, class_id=cat_id)
                    
                    # Crop the text from the text image
                    text_bbox = text_annotation['bbox']
                    text = crop_signature(text_image, text_bbox)

                    # Composite the text on the check
                    filled_check, new_bbox = fill_check_field(filled_check, text,
                                                                 target_bbox, pen_color,
                                                                 random_position=False)
                    

                # Add the new image and annotation to the new dataset
                new_annotation = annotation.copy()
                new_annotation['id'] = len(forged_checks_annotations) + 1
                new_annotation['image_id'] = new_id
                new_annotation['category_id'] = cat_id
                new_annotation['bbox'] = new_bbox
                new_annotation['area'] = new_bbox[2] * new_bbox[3]
                new_annotation['attributes']['ink_color'] = pen_color
                new_annotation['attributes']['ink_name'] = pen_name
                forged_checks_annotations.append(new_annotation)

            # Add the new image info to the new dataset
            new_image_name = f'{bank_check_id}_{signature_id}_{i}.jpg'
            new_image_info = image_info.copy()
            new_image_info['id'] = new_id
            new_image_info['file_name'] = new_image_name
            new_image_info['width'] = filled_check.shape[1]
            new_image_info['height'] = filled_check.shape[0]
            forged_checks_images.append(new_image_info)
            new_id += 1

            # Save the new image
            cv2.imwrite(f'{output_images_path}/{new_image_name}', filled_check)

forged_checks_data['images'] = forged_checks_images
forged_checks_data['annotations'] = forged_checks_annotations

# Save the new dataset
with open(f'{output_dir}/labels.json', 'w') as file:
    json.dump(forged_checks_data, file)
