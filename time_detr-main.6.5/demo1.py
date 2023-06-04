import os
import cv2
import xml.etree.ElementTree as ET


def draw_bounding_boxes(image_path, xml_path, output_folder):
    image = cv2.imread(image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # 保存绘制了边界框的图像
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)


image_folder = r'/home/zcs/code/deep-learning-for-image-processing-master/predict_demo/output/res_ciou_da'
xml_folder = r'/home/zcs/code/deep-learning-for-image-processing-master/predict_demo/demo_xml'
output_folder = r'/home/zcs/code/deep-learning-for-image-processing-master/predict_demo/final_output/res50_ciou_da'

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]
    xml_file = f'{image_name}.xml'

    if xml_file in os.listdir(xml_folder):
        image_path = os.path.join(image_folder, image_file)
        xml_path = os.path.join(xml_folder, xml_file)

        draw_bounding_boxes(image_path, xml_path, output_folder)
    else:
        print(f'XML file not found for image: {image_file}')
