import csv
import json

# Open the CSV file containing the bounding box and class label information
with open(r':\path\train.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    # Open a new text file for writing the converted bounding box coordinates
    with open('/path', 'w') as txtfile:

        # Iterate through the rows in the CSV file
        for row in reader:

            # Construct the image path using the filename column
            filename = row['filename']
            image_path = f'D:/Alex/{filename}'

            # Extract the bounding box coordinates from the JSON string in the region_shape_attributes column
            region_attributes = json.loads(row['region_shape_attributes'])
            x_min = int(float(region_attributes['x']))
            y_min = int(float(region_attributes['y']))
            width = float(region_attributes['width'])
            height = float(region_attributes['height'])

            # Convert the bounding box coordinates to (x_min, y_min, x_max, y_max) format

            x_max = int(x_min + width)
            y_max = int(y_min + height)

            # Extract the class ID from the JSON string in the region_attributes column
            region_attributes = json.loads(row['region_attributes'])
            class_id = int(region_attributes['class id'])

            # Write the converted bounding box coordinates and class ID to the text file in the YOLO format
            txtfile.write(f'{image_path},{x_min},{y_min},{x_max},{y_max},{class_id}\n')
