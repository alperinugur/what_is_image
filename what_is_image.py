import os
import io
import openai
from google.cloud import vision
import re
from utils import FILES,initialize, select_file

GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

def chatGPTimageResult( prompt, Language = "English"):

    myTable = VisionToTable(prompt)
    # print (f"\n\nMY TABLE:\n\n{myTable}")   #enable to see the table
    imageReqLine = [{
        "role": "system", 
        "content": "You are a predictor which analysis text contents about a picture and sends back the best possible results. You give prompt answers and you do not mention about the table given to you. You just type in what you understand from the input material"
        }]
    reptr=""
    if Language != "English":
        reptr = f"Please provide the answer fully in {Language}"
    
    newline =  imageReqLine+[{"role": "user", "content": f"Here is the probabilities about a picture as a table:\n\n{myTable}\n\nPlease type down what is in the picture considering this table. Do not mention the table or score numbers, just give an answer like an author. {reptr} "}]
    # print (newline)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=newline,
            max_tokens=2500,
            temperature = 0.4
        )
        ChatGPT_reply = response["choices"][0]["message"]["content"]
        ln1,ln2,ln3 = response["usage"]["prompt_tokens"],response["usage"]["completion_tokens"],response["usage"]["total_tokens"]
        tokeninfo = f" - [Tokens: {ln1}-{ln2}-{ln3}]"
        #tokeninfo = ""
        return (f"{ChatGPT_reply}")
    except Exception as e:
        print (f"Exception:\n{e}\n")

        return (e)

def Interrogate_Image(image_file_path):

    client = vision.ImageAnnotatorClient()

    with io.open(image_file_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.annotate_image({
    'image': image,
    'features': [{'type_': vision.Feature.Type.LABEL_DETECTION}]
    })
    return (str(response))

def VisionToTable(content):
    blocks = content.split('label_annotations')
    # Parse each block and append it to a list
    data = []
    mystr="LABELS:description,score,topicality\n"
    for block in blocks:
        if block.strip() != '':
            mystr= mystr+parse_block2(block)
    # Convert the list of dictionaries into a DataFrame
    #print (f"Deneme Str: {mystr}")
    return(mystr)

def parse_block2(block):
    """
    Parses a block of text and returns a dictionary containing the data.
    """
    lines = block.split('\n')

    TheStr="DATA:"

    for line in lines:
        line = line.strip()
        match = re.match(r'description: "(.*)"', line)
        if match:
            TheStr = TheStr+ match.group(1) + ","
        else:
            match = re.match(r'score: (\d+\.\d+)', line)
            if match:
                # Convert to float and format with two decimal places
                TheStr = TheStr+ format(float(match.group(1)), '.2f') + ","
            else:
                match = re.match(r'topicality: (\d+\.\d+)', line)
                if match:
                    TheStr = TheStr+ format(float(match.group(1)), '.2f') + "\n"
        
    return TheStr


# atexit.register(cleanup)

if __name__ == '__main__':
    initialize()
    file = select_file()
    whatisinimage = Interrogate_Image(file)[0:2000]
    # print (whatisinimage)  #enable this line to see what is going on
    imageContains= chatGPTimageResult(whatisinimage,Language="English")
    print (imageContains)
