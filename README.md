# projectAI
Help for dyslexic patients in recognizing reversed letters.

The purpose of this project is to help dyslexic patients locate reversed letters written in mirror script.
These people have no way of knowing whether they wrote the letter correctly. 

Today there is no solution that helps them write properly by hand, but only technological solutions that include computerized writing. 
In order for them to be able to write everyday things by hand like other people, we developed this project.

The user must upload a photo of a handwritten text and the text will be returned to him with the reverse letters marked.
In the project we used an existing model from Hugging Face whose purpose is to convert text from a handwritten image to computer text. 
With the help of this model we were able to create a comparison between three texts. 
One of the text from the original model, the second of images of letters each of which was uploaded to the model and the letter was returned from it, 
and the third of the reverse image, and reading each image of a letter from it. 
Comparing the three, helped to locate the opposite letters. After marking those letters, we returned to the user.
