import os
import xml.etree.ElementTree as ET
import cv2

# PPE classes
CLASSES = ['Coverall', 'Gloves', 'Goggles', 'Mask', 'Face_Shield'] 

def convert_voc_to_yolo(xml_folder, image_folder, output_label_folder):
    os.makedirs(output_label_folder, exist_ok=True)

    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, file)
            image_file = file.replace(".xml", ".jpg") 
            image_path = os.path.join(image_folder, image_file)

            if not os.path.exists(image_path):
                print(f"[WARNING] Skipping {image_file}: image not found.")
                continue

            img = cv2.imread(image_path)
            h, w, _ = img.shape

            tree = ET.parse(xml_path)
            root = tree.getroot()

            yolo_labels = []

            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls not in CLASSES:
                    continue
                cls_id = CLASSES.index(cls)

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                x_center = (xmin + xmax) / 2 / w
                y_center = (ymin + ymax) / 2 / h
                bbox_width = (xmax - xmin) / w
                bbox_height = (ymax - ymin) / h

                yolo_labels.append(f"{cls_id} {x_center} {y_center} {bbox_width} {bbox_height}")

            label_file = os.path.join(output_label_folder, file.replace(".xml", ".txt"))
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_labels))

    print(f"Done converting: {xml_folder}")

if __name__ == "__main__":
    # train set
    convert_voc_to_yolo(
        xml_folder=r"C:\Users\Sonali Maharana\Desktop\dataset\train",
        image_folder=r"C:\Users\Sonali Maharana\Desktop\dataset\train",
        output_label_folder=r"C:\Users\Sonali Maharana\Desktop\dataset\labels\train"
    )

    # test set (used as val in YOLO)
    convert_voc_to_yolo(
        xml_folder=r"C:\Users\Sonali Maharana\Desktop\dataset\test",
        image_folder=r"C:\Users\Sonali Maharana\Desktop\dataset\test",
        output_label_folder=r"C:\Users\Sonali Maharana\Desktop\dataset\labels\val"
    )