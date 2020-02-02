from PIL import Image
from google.cloud import vision
import io

global img_params
img_params = [0,0,0,0]

global compression
compression = 5

def thresh(color_tup):
    thresh_red_upper = 170
    thresh_green_upper = 170
    thresh_blue_upper = 170
    if(color_tup[0]<thresh_red_upper and color_tup[1]<thresh_green_upper and color_tup[2]<thresh_blue_upper):
        return True
    return False

def thresh2(color_tup):
    thresh_green_upper = 50
    thresh_blue_upper = 50
    if(color_tup[1]<thresh_green_upper and color_tup[2]<thresh_blue_upper):
        return True
    return False

def recur_bird(x,y,new_color,old_color,pix,width2,height2):
    if x<img_params[0]:
        img_params[0] = x
    if x>img_params[1]:
        img_params[1] = x
    if y<img_params[2]:
        img_params[2] = y
    if y>img_params[3]:
        img_params[3] = y

    if(pix[x,y]!=old_color or x<0 or y<0 or x>=width2-1 or y>=height2-1):
        return 0

    pix[x,y]=new_color #yes this pixel has been checked
    #bird_size+=1

    #recursion for all 4 directions
    return 1+recur_bird(x+1, y,new_color,old_color,pix,width2,height2)+recur_bird(x-1, y,new_color,old_color,pix,width2,height2)+recur_bird(x, y+1,new_color,old_color,pix,width2,height2)+recur_bird(x, y-1,new_color,old_color,pix,width2,height2)


# #increase image contrast
def recolor_image(current_image):
    im = Image.open(current_image)

    size_of_image = im.size
    width = size_of_image[0]
    height = size_of_image[1]
    while width * height > 25000:
        im.thumbnail((width//2,height//2))
        width, height = im.size
        #print(width*height)
    im.save('sample_thumbnail.jpg')

    im2 = Image.open('sample_thumbnail.jpg')

    pix = im2.load()

    size_of_image = im2.size
    width2 = size_of_image[0]
    height2 = size_of_image[1]

    #recolor the image to black and white
    for i in range(width2):
        for j in range(height2):
            color_tup = pix[i, j]
            if(thresh(color_tup)==True):
                pix[i,j] = (0,0,0)  # Set the RGBA Value of the image (tuple)
            else:
                pix[i,j] = (255,255,255)

    count = 1

    all_the_images = []

    for x in range(width2):
        for y in range(height2):
            if(pix[x,y]==(0,0,0)):
                #print(x,y)
                img_params[0] = x
                img_params[1] = x
                img_params[2] = y
                img_params[3] = y
                num_pixels = recur_bird(x,y,(255,0,0),(0,0,0), pix, width2, height2)
                #print("Number of pixels: ",num_pixels)
                #print(img_params)

                if num_pixels > 100:
                #crop the image here
                    crop_img = im2.crop((img_params[0],img_params[2],img_params[1],img_params[3]))

                    the_string = "cropped_img"+str(count)+".jpg"
                    crop_img.save(the_string)
                    all_the_images.append(the_string)
                    count+=1


    ar_img = []
    for clare in all_the_images:
        #print(clare)
        im = Image.open(clare)
        size_of_image = im.size
        width = size_of_image[0]
        height = size_of_image[1]
        pix = im.load()
        #print("HIII")

        #recolor the image to black and white
        for i in range(width):
            for j in range(height):
                color_tup = pix[i, j]
                if(thresh2(color_tup)==True):
                    pix[i,j] = (0,0,0)  # Set the RGBA Value of the image (tuple)
                else:
                    pix[i,j] = (255,255,255)
        im.save(clare)
        ar_img.append(im)

    return ar_img

def classify(rgb_tuple):
    colors = {"white": (255, 255, 255),
              "black" : (0, 0, 0),
              }

    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key = distances.get)

    return color

def compare_imgs(im1, im2):
    #im1 = Image.open(fp1)
    #im2 = Image.open(fp2)

    im1 = im1.resize((20, 30))
    im2 = im2.resize((20, 30))

    num_match = 0
    for x in range(20):
        for y in range(30):
            if classify(im1.getpixel((x, y))) == classify(im2.getpixel((x, y))):
                num_match += 1

    return num_match / (20 * 30)

def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    symbols = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            #print(block)
            for paragraph in block.paragraphs:
                #print(paragraph)
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    #print(word_text)
                    #print('Word text: {} (confidence: {})'.format(word_text, word.confidence))
                    for symbol in word.symbols:
                        symbols.append(symbol.text)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return symbols

def tag_imgs(text, img_array):
    tag_dict = {}
    for x in range(len(text)):
        tag_dict[text[x]] = img_array[x]

    return tag_dict

def final_compare_imgs(fp1, fp2):
    arr_imgs1 = recolor_image(fp1)
    text1 = detect_document(fp1)
    dicted1 = tag_imgs(text1, arr_imgs1)
    arr_imgs2 = recolor_image(fp2)
    text2 = detect_document(fp2)
    dicted2 = tag_imgs(text2, arr_imgs2)

    cum_pct = 0
    ct = 0
    for k in dicted1.keys():
        if k in dicted2:
            cum_pct += compare_imgs(dicted1[k], dicted2[k])
            ct += 1

    pct_similar = cum_pct / ct
    if pct_similar > 0.75:
        return (True, pct_similar)
    return (False, pct_similar)

if __name__ == '__main__':
    print(final_compare_imgs('IMG_1810.jpg', 'IMG_1814.jpg'))
    print(final_compare_imgs('IMG_1810.jpg', 'IMG_1813.jpg'))

    # arr_imgs1 = recolor_image('IMG_1814.jpg')
    # #print(arr_imgs)
    # text1 = detect_document('IMG_1814.jpg')
    # print(text1)
    # dicted1 = tag_imgs(text1, arr_imgs1)
    #
    # #dicted['A'].show()
    # #print('PERCENT EQUAL:', compare_imgs(dicted['A'], dicted['A']))
    #
    # # arr_imgs2 = recolor_image('IMG_1813.jpg')
    # # text2 = detect_document('IMG_1813.jpg')
    # # #print(text2)
    # # dicted2 = tag_imgs(text2, arr_imgs2)
    # #
    # # cum_pct = 0
    # # ct = 0
    # # for k in dicted1.keys():
    # #     if k in dicted2:
    # #         cum_pct += compare_imgs(dicted1[k], dicted2[k])
    # #         ct += 1
    # #
    # # print('AVERAGE SIMILARITY:', cum_pct / ct)
