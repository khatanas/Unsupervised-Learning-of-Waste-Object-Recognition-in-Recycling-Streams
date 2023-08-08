from helper.common_libraries import cv2,json

sortedUnique = lambda my_list: sorted(list(set(my_list)))
flattenList = lambda my_list: [item for sublist in my_list for item in sublist]

#****************** cv2 **************************
def imreadRGB(path_file_image):
    image = cv2.imread(path_file_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def imreadGray(path_file_image):
    image = cv2.imread(path_file_image, cv2.IMREAD_GRAYSCALE)
    return image
#******************* JSON **************************
def writeJson(path_file_json, json_file):
    '''
    Write json_file at path_file_json
    '''
    with open(path_file_json, 'w') as f:
        json.dump(json_file,f)


def readJson(path_file_json):
    '''
    Load json_file located at path_file_json
    '''
    with open(path_file_json, 'r') as f:
        json_file = json.load(f)
    return json_file

