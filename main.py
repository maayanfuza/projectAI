from tkinter import *
from tkinter import filedialog
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
print(torch.__version__)

def origionModel(image):
    """
     The original model from hugging face
     :param image: get the image file with the incorrect text
     :type image: png/ jpg file
     :return: text that written in the image
     :rtype: string
     """
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(image)
    if generated_text.endswith('.'):
        generated_text = generated_text[:-1]
    print("origionModel:", generated_text)
    return generated_text


import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

def devide2letters(image_cv2):
    """
     get an image file, devide it into small images of the letters inside,
     and reading them using the original model
     :param image_cv2: get the image file with the text to read
     :type image_cv2: png/ jpg file
     :return: text that written in each letter in the image
     :rtype: string
     """
  #image_cv2 = cv2.imread(image_path)

  # Convert the image to grayscale
  gray = cv2.cvtColor(np.array(image_cv2), cv2.COLOR_BGR2GRAY)

  # Apply thresholding to binarize the image
  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # Find contours
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Sort contours by x-coordinate
  contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

  entire_word=""  #########i added

  # Iterate through each sorted contour
  for i, contour in enumerate(contours_sorted):
      # Get bounding box for the contour
      x, y, w, h = cv2.boundingRect(contour)

      # Crop the bounding box region from the original image
      letter_image_cv2 = np.array(image_cv2)[y:y+h, x:x+w]
      letter_image_pil = Image.fromarray(letter_image_cv2)

      # Save the cropped region as a separate image
      letter_image_pil.save(f'C:/Users/IMOE001/Desktop/לימודים/שנה ג/בינה מלאכותית/קובץ אותיות/{i}.jpg')

      # Load the cropped image
      image = Image.open(f'C:/Users/IMOE001/Desktop/לימודים/שנה ג/בינה מלאכותית/קובץ אותיות/{i}.jpg').convert("RGB")

      processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
      model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
      pixel_values = processor(images=image, return_tensors="pt").pixel_values

      generated_ids = model.generate(pixel_values)
      generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
      if(len(generated_text) > 1):
        generated_text = generated_text[0]
      #print(generated_text)

      entire_word += generated_text
      #display(image)
      #print(generated_text)
  print("devide2letters:",entire_word)
  return entire_word


import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

def flip_image(image_cv2):
    """
     get an image file, flip it into its mirror image, devide it to letters
     and reading them using the original model
     :param image_cv2: get the image file with the text to read
     :type image_cv2: png/ jpg file
     :return: text that written in each letter in the image, the entire text from the image
     :rtype: string
     """

  #image_cv2 = cv2.imread(image_path)

  # Flip the image horizontally
  flipped_image = cv2.flip(np.array(image_cv2), 1)

  # Convert the flipped image to grayscale
  gray = cv2.cvtColor(np.array(flipped_image), cv2.COLOR_BGR2GRAY)

  # Apply thresholding to binarize the image
  _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # Find contours
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Sort contours by x-coordinate
  contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

  entire_word=""  #########i added


  # Iterate through each sorted contour
  for i, contour in enumerate(contours_sorted):
      # Get bounding box for the contour
      x, y, w, h = cv2.boundingRect(contour)

      # Crop the bounding box region from the original image
      letter_image_cv2 = np.array(flipped_image)[y:y+h, x:x+w]
      letter_image_pil = Image.fromarray(letter_image_cv2)

      # Save the cropped region as a separate image
      letter_image_pil.save(f'C:/Users/IMOE001/Desktop/לימודים/שנה ג/בינה מלאכותית/קובץ אותיות/{i}.jpg')

      # Load the cropped image
      image = Image.open(f'C:/Users/IMOE001/Desktop/לימודים/שנה ג/בינה מלאכותית/קובץ אותיות/{i}.jpg').convert("RGB")

      processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
      model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
      pixel_values = processor(images=image, return_tensors="pt").pixel_values

      generated_ids = model.generate(pixel_values)
      generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
      if(len(generated_text) > 1):
        generated_text = generated_text[0]

      entire_word += generated_text #########i added
      #display(image)
      #print(generated_text)
  print("flip_image:",entire_word)
  return entire_word



def compare_strings(text1, text2, text3):
     """
     compare three texts, one from reading it from the entire text, the other from the devided letter,
     and the third from the flipped image devided to letters.
     removing all the spaces before
     :param text1: the text from the original model
     :type text1: string
     :param text2: the text from the devided letters from the original model
     :type text2: string
     :param text1: the text from the flipped image from the original model
     :type text1: string
     :return error_list: list with the indexs of the flipped words in the text
     :rtype error_list: list
     :return: text that written in the entire image
     :rtype: string
     """
  tmp_text1 = text1.replace(" ","")
  text3 = text3[::-1]
  new_text = ""
  error_list = []

  strings = [tmp_text1,text2,text3]
  shortest_string = min(strings, key=len)

  for i in range(len(shortest_string)):
    if tmp_text1[i] == text2[i]:
      new_text += tmp_text1[i]
    else:
      if text3[i].lower() == tmp_text1[i].lower():
        new_text += text3[i]
        error_list.append(i)
      else:
        new_text += tmp_text1[i]

  print(new_text)
  print(error_list)
  return error_list,tmp_text1


def emphasize_letters(text, indexes):
    """
     takes the text, and bold the letters that their location in the indexs list
     :param text: the text from the original model
     :type text: string
     :param indexes: list with the indexs of the flipped words in the text
     :type indexes: list
     :return result: text that written in the entire image, the letters in the index's location are bold
     :rtype result: string
     """
    result = ""
    for i, char in enumerate(text):
        if i in indexes:
            result += char + "\u0332"  # הוספת הדגשה באמצעות הספרה השלישית של קוד ה-Unicode
        else:
            result += char
    return result



def add_spaces_change_index(string_without_spaces, string_with_spaces,indexes):
    """
     takes the text, and add spaces when needed, change the indexs list respondedly
     :param string_without_spaces: the text with no spaces
     :type string_without_spaces: string
     :param string_with_spaces: text with spaces
     :type string_with_spaces: string
     :param indexes: list with the indexs of the flipped words in the text
     :type indexes: list
     :return string_with_spaces_added: text that with spaces
     :rtype string_with_spaces_added: string
     :return indexes: the indexs list, changed
     :rtype indexes: list
     """
    string_with_spaces_added = ""
    index_without_spaces = 0
    for char in string_with_spaces:
        if index_without_spaces < len(string_with_spaces):
          if char==" ":
            string_with_spaces_added += " "
            index_without_spaces += 1
            for i in range(len(indexes)):
              if indexes[i]>=(index_without_spaces-1):
                indexes[i]+=1
          else:
            string_with_spaces_added += char
            index_without_spaces += 1
    return string_with_spaces_added,indexes



def model(image):
    """
  get an image, return the text written in the image, the flipped letters bold
  :param image: the image the user want to get feedback on
  :type image: png/ jpg file
  :return: text that written in the entire image, the flipped letters bold
  :rtype: string
  """
  text1=origionModel(image)
  text2=devide2letters(image)
  text3=flip_image(image)

  errorList,tmp_text1=compare_strings(text1,text2,text3)
  string_without_spaces = tmp_text1
  string_with_spaces = text1
  result,index = add_spaces_change_index(string_without_spaces, string_with_spaces,errorList)
  return (emphasize_letters(result,index))

import gradio as gr

def demo(image):
    """
  get an image, and send it to the model func, returns the text with bold lettes
  :param image: the image the user want to get feedback on
  :type image: png/ jpg file
  :return: text that written in the entire image, the flipped letters bold
  :rtype: string
  """
    def greet(image):
        return model(image)

    demo = gr.Interface(fn=greet, inputs="image", outputs="text")

    demo.launch(share=True)



def open_image():
    """
      open the image for the app
      :return: none
      :rtype: none
      """
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    result_text = demo(image)
    result_label.config(text=result_text)


def main():
    app = Tk()
    app.title("Upload and Process Image")
    app.configure(bg="black")
    app.geometry('1000x500')

    upload_button = Button(app, text="Upload Image", command=open_image)
    upload_button.pack()

    result_label = Label(app, text="", bg="black", fg="white", font=("Arial", 12))
    result_label.pack()

    app.mainloop()

if __name__ == "__main__":
    main()