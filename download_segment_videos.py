import os
import sys
import getopt
import pytube
import shutil
import numpy as np
from pytube import YouTube

def video_segmentation(orginal_video_name,segment_file,main_folder_path):

	file_segments_data = np.genfromtxt(segment_file,dtype=[('f8'),('f8'),('S50')],delimiter='\t')
	
	if file_segments_data.shape == ():
		file_segments_data = np.atleast_1d(file_segments_data)

	#Create folder for segments
	gral_segments_path = os.path.join(main_folder_path,'Segments')
	if not os.path.isdir(gral_segments_path):
		os.mkdir(gral_segments_path)
	
	counter_enter_out = 0; #We know that the segments are in order, so we need to check they match and no match of the file names then our algorithm is more efficient
	for k in range(len(file_segments_data['f2'])):

		if orginal_video_name[:-4] in file_segments_data['f2'][k].decode('utf-8'):

			if counter_enter_out == 0:
				counter_enter_out = 1;

			video_input = os.path.join(main_folder_path,orginal_video_name);

			#Create folder for segment
			segments_path = os.path.join(gral_segments_path,file_segments_data['f2'][k].decode('utf-8')[:-4])
			if not os.path.isdir(segments_path):
				os.mkdir(segments_path)

			segment_output = os.path.join(segments_path,file_segments_data['f2'][k].decode('utf-8'))

			ti = file_segments_data['f0'][k]
			dt = file_segments_data['f1'][k] - file_segments_data['f0'][k]

			os.system("ffmpeg -loglevel panic -i %s -strict -2 -break_non_keyframes 1/0 -ss %s -t %s %s" %\
				(video_input,str(ti),str(dt),segment_output))

			#Getting Audio
			audio_path = os.path.join(segments_path,'Audio')
			if not os.path.isdir(audio_path):
				os.mkdir(audio_path)

			os.system("ffmpeg -loglevel panic -i %s %s" % (segment_output,os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + "O.wav")))

			os.system("sox %s %s remix 1" % (os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + "O.wav"),os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + "S.wav")))

			os.system("ffmpeg -loglevel panic -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + "S.wav"),os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + ".wav")))

			os.remove(os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + "O.wav"))
			os.remove(os.path.join(audio_path,file_segments_data['f2'][k].decode('utf-8')[:-4] + "S.wav"))

			if counter_enter_out == 1:
				break;

	return 0;


def download_videos(link_file,segment_file,main_folder,segments_path):

	#Read text file
	file_links_data = np.genfromtxt(link_file,dtype=[('S30'),('S50')],delimiter='\t')#f0:name database, f1: link

	if file_links_data.shape == ():
		file_links_data = np.atleast_1d(file_links_data)

	for k in range(len(file_links_data['f1'])):
		link = file_links_data['f1'][k].decode('utf-8');
		print(link)

		#Create folder to save the original video
		main_folder_video = os.path.join(main_folder,file_links_data['f0'][k].decode('utf-8')[:-4]) 
		if not os.path.isdir(main_folder_video):
			os.mkdir(main_folder_video)

		#Download video, high quality stream, mp4
		try:
			#Downloading the video
			yt_video = YouTube(link)
			yt_video.streams.filter(progressive=True, \
				file_extension='mp4').order_by('resolution')[-1].download(output_path=main_folder_video,filename=file_links_data['f0'][k].decode('utf-8')[:-4]);

			#Segmenting the video
			video_segmentation(file_links_data['f0'][k].decode('utf-8'),segment_file,main_folder_video)
		
		except KeyboardInterrupt:
			  print("Good bye!")
			  exit(0)		
		except:
			print("The video is not available. Continue with the next video.")

def main(argv):


	#Reading command line	
	link_file = ''
	segment_file = ''
	download_path = ''
	segments_path = ''
	options = "h:l:s:d:p:"
	longOptions = ["help","links","segment_data","path_savedata"]
	try:
		opts, args = getopt.getopt(argv,options,longOptions)

	except getopt.GetoptError:
		print('download_segment_videos.py -l <linksFile> -s <segmentsFile> -p <saveDataPath>')
		sys.exit(2)

		
	for opt, arg in opts:

		if opt in ("-h","--help"):
			print('download_segment_videos.py -l <linksFile> -s <segmentsFile> -p <saveDataPath>')
			sys.exit(2)
		elif opt in ("-l","--links"):
			link_file = arg
		elif opt in ("-s","--segment_data"):
			segment_file = arg
		elif opt in ("-p","--path_savedata"):
			savedata_path = arg

	#Create main MSP-Face folder
	main_folder = os.path.join(savedata_path,'MSP-Face')
	if not os.path.isdir(main_folder):
		os.mkdir(main_folder)

	#Download videos
	download_videos(link_file,segment_file,main_folder,segments_path);

if __name__ == "__main__":

	main(sys.argv[1:]);
