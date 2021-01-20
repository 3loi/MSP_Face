# import the necessary packages
import imutils
import cv2
from imutils.face_utils import FaceAligner
import dlib

class FaceAlignerPreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter
		self.shape_preditor_path = "./shape_predictor_68_face_landmarks.dat"
		self.predictor = dlib.shape_predictor(self.shape_preditor_path)
		self.facealigner = FaceAligner(self.predictor, desiredFaceWidth=self.width)
		self.detector = dlib.get_frontal_face_detector()


	def preprocess(self, image):
		image = imutils.resize(image, width=800)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		rects = self.detector(gray, 2)
		if len(rects) < 1:
			print("unable to locate face")
		for rect in rects:
			return self.facealigner.align(image, gray, rect)
 			
		return None
