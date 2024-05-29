from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import requests

import numpy as np

from determiner import validate, trainModel, trainGeneratorModel, generateImage, INPUTS_PATH, TARGET_PATH

import base64

import numpy as np
import tensorflow as tf
import PIL.Image as Image
from io import BytesIO
import os
import json

hostName = "127.0.0.1"
serverPort = 80

def downloadImageFrom(url):
    res = requests.get(url)

    if not res.ok:
        print(res)
        print("Image download response error")
        return
    
    print(len(res.content))
    print(str(res.elapsed.total_seconds()) + " seconds elapsed on request")
    return res.content
def logUrl(fileName, url):
    data = {}
    with open(TARGET_PATH+"/urlLog.json", 'rt') as file:
        data = file.read()

    data = json.loads(data)
    data[fileName] = url
    with open(TARGET_PATH+"/urlLog.json", 'wt') as file:
        file.write(json.dumps(data))

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        urlData = urlparse(self.path)
        parsedQuery = parse_qs(urlData.query)
        
        
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        #self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        print(urlData)
        if (urlData.path == '/' and urlData.query == ''):
            with open("index.html","rb") as file:
                self.wfile.write(file.read())
        elif (urlData.path == '/debug'):
            with open("debug.html","rb") as file:
                self.wfile.write(file.read())
        
        if 'get_debug_length' in parsedQuery:
            target = parsedQuery['target'][0]
            path = "DebugImages/"+target
            
            self.wfile.write(bytes(str(len(os.listdir(path))),'utf-8'))
        if 'get_debug_image' in parsedQuery:
            target = parsedQuery['target'][0]
            # data = [[],[]]
            # for file in os.listdir("DebugImages/0"):
            #     target = "DebugImages/0/" + file
            #     with open(target,"rb") as img:
            #         data[0].append(img.read())
            
            # for file in os.listdir("DebugImages/1"):
            #     target = "DebugImages/1/" + file
            #     with open(target,"rb") as img:
            #         data[1].append(img.read())
            
            path = "DebugImages/"+target
            if os.path.exists(path):
                with open(path,"rb") as img:
                    self.wfile.write(img.read())
            

            # print(data[0][0])
            # self.wfile.write(data[0][0])
            # self.wfile.write(bytes(json.dumps(data),'utf-8'))
            
                
        if 'recognize' in parsedQuery:
            target = parsedQuery['src'][0]
            img = downloadImageFrom(target)
            
            img = Image.open(BytesIO(img))
            img = img.convert("RGB")
            img = img.resize((400,400))
            arr = np.array(img, dtype='float16')
            prediction = validate(np.array([arr])/255)

            # print((prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3],prediction[0][4]))
            
            targetPath = ""
            if (prediction[0][0][0] > prediction[0][0][1]):
                targetPath = "0/img"+str(len(os.listdir("DebugImages/0")))+".jpg"
            else:
                targetPath = "1/img"+str(len(os.listdir("DebugImages/1")))+".jpg"
            img.save("DebugImages/"+targetPath)

            data = {}
            with open("DebugImages/log.json", 'rt') as file:
                data = file.read()

            data = json.loads(data)
            data[targetPath] = prediction[0][0].tolist()
            with open("DebugImages/log.json", 'wt') as file:
                file.write(json.dumps(data))
            
            self.wfile.write(bytes(json.dumps({"done":1, "recognition":prediction[0][0].tolist()}),"utf-8"))
        elif 'transfer' in parsedQuery:
            From = parsedQuery['from'][0]
            To = parsedQuery['to'][0]

            target = "Inputs/"+To+"/img"+str(len(os.listdir("Inputs/"+To)))+".jpg"
            os.rename(From,target)

            ppp = "DebugImages/"+From.split('/')[1]
            os.mkdir(ppp+"/new")
            for img in os.listdir(ppp):
                if img == 'new': continue
                file = ppp+"/"+img
                os.rename(file,ppp+"/new/"+img)
            
            ind = 0
            for img in os.listdir(ppp+"/new"):
                file = ppp+"/new/"+img
                os.rename(file,ppp+"/img"+str(ind)+".jpg")
                ind += 1

            os.rmdir(ppp+"/new")

            self.wfile.write(bytes("translated",'utf-8'))
        elif 'score' in parsedQuery:
            target = parsedQuery['src'][0]
            score = int(parsedQuery['score'][0])

            print(f"Target: {target}")
            print(f"Score: {score}")

            img = downloadImageFrom(target)
            img = Image.open(BytesIO(img))

            folderPath = f"{INPUTS_PATH}/class_{score+1}"
            filesCount = len(os.listdir(folderPath))
            fileName = f"{folderPath}/img{filesCount}.jpg"
            img.save(fileName)

            logUrl(fileName,target)

            self.wfile.write(bytes(json.dumps({"done":1,"file-name":fileName}),"utf-8"))
        elif 'train' in parsedQuery:
            epochsCount = parsedQuery['epochsCount'][0]
            if ('learningRate' in parsedQuery):
                learningRate = parsedQuery['learningRate'][0]
            else:
                learningRate = 1e-4
            l1 = 0 if (not('l1' in parsedQuery)) else parsedQuery['l1'][0]
            l2 = 0 if (not('l2' in parsedQuery)) else parsedQuery['l2'][0]
            if ('continueTrain' in parsedQuery):
                continueTrain = parsedQuery['continueTrain'][0]

            trainModel(int(epochsCount), float(learningRate), float(l1), float(l2), bool(int(continueTrain)))

            self.wfile.write(bytes("loss: "+" | accuracy: ","utf-8"))
        elif 'train_generator' in parsedQuery:
            epochsCount = parsedQuery['epochsCount'][0]
            if ('learningRate' in parsedQuery):
                learningRate = parsedQuery['learningRate'][0]
            else:
                learningRate = 1e-4
                learningRate = 1e-4

            l1 = 0 if (not('l1' in parsedQuery)) else parsedQuery['l1'][0]
            l2 = 0 if (not('l2' in parsedQuery)) else parsedQuery['l2'][0]
            if ('continueTrain' in parsedQuery):
                continueTrain = parsedQuery['continueTrain'][0]

            trainGeneratorModel(int(epochsCount), float(learningRate), float(l1), float(l2), bool(int(continueTrain)))

            self.wfile.write(bytes("Train over","utf-8"))
        elif 'generate_image' in parsedQuery:
            target = parsedQuery['target'][0]

            img = generateImage(int(target)+1)
            #self.wfile.write(img.tobytes())
            img = Image.fromarray(img)
            img.save("GenerateImages/test"+str(len(os.listdir("GenerateImages")))+".png")
            # img.show()
            self.wfile.write(bytes("test","utf-8"))

        







            

            




if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")