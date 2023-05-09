import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticMode=False, maxFaces=1, redefineLm= False, minDetectionCon=0.5, minTrackCon=0.5, thickness=5, circle_radius=5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.redefineLm = redefineLm
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.thickness = thickness
        self.circle_radius = circle_radius

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.redefineLm,
                                                 self.minDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=self.thickness, circle_radius=self.circle_radius)

    def findFaceMesh(self,img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                        self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x,y])
                faces.append(face)
        return img, faces
    
def main():
    cap = cv2.VideoCapture("videos/Girl_Phone.mov")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        
        scale_percent = 25  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        imgToShow = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(imgToShow, f'FPS {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
                    3, (0,255,0),2)
        
        cv2.imshow("Image", imgToShow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

if __name__ == "__main__":
    main()