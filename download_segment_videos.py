import os
import sys
import getopt
import pytube
import shutil
import numpy as np
from pytube import YouTube

def video_segmentation(orginal_video_name,segment_file,download_path,segments_path):

	file_segments_data = np.genfromtxt(segment_file,dtype=[('f8'),('f8'),('S50')],delimiter='\t')
	
	if file_segments_data.shape == ():
		file_segments_data = np.atleast_1d(file_segments_data)
	
	counter_enter_out = 0; #We know that the segments are in order, so we need to check the matc and no match of the file names then our algorithm is more efficient
	for k in range(len(file_segments_data['f2'])):

		if orginal_video_name[:-4] in file_segments_data['f2'][k].decode('utf-8'):

			if counter_enter_out == 0:
				counter_enter_out = 1;

			video_input = os.path.join(download_path,orginal_video_name);
			segment_output = os.path.join(segments_path,file_segments_data['f2'][k].decode('utf-8'))

			ti = file_segments_data['f0'][k]
			dt = file_segments_data['f1'][k] - file_segments_data['f0'][k]

			os.system("ffmpeg -loglevel panic -i %s -strict -2 -break_non_keyframes 1/0 -ss %s -t %s %s" %\
				(video_input,str(ti),str(dt),segment_output))

		else:
			if counter_enter_out == 1:
				break;

	return 0;


def download_videos(link_file,segment_file,download_path,segments_path):

	#Read text file
	file_links_data = np.genfromtxt(link_file,dtype=[('S30'),('S50')],delimiter='\t')#f0:name database, f1: link

	if file_links_data.shape == ():
		file_links_data = np.atleast_1d(file_links_data)

	for k in range(len(file_links_data['f1'])):
		link = file_links_data['f1'][k].decode('utf-8');
		print(link)

		#Download video, high quality stream, mp4
		try:
			#Downloading the video
			yt_video = YouTube(link)
			yt_video.streams.filter(progressive=True, \
				file_extension='mp4').order_by('resolution')[-1].download(output_path=download_path,filename=file_links_data['f0'][k].decode('utf-8')[:-4]);

			#Segmenting the video
			video_segmentation(file_links_data['f0'][k].decode('utf-8'),segment_file,download_path,segments_path)
				
		except:
			print("The video is not available. Continue with the next video.")

def main(argv):


	#Reading command line	
	link_file = ''
	segment_file = ''
	download_path = ''
	segments_path = ''
	options = "h:l:s:d:p:"
	longOptions = ["help","links","segment_data","downVideos","downSegments"]
	try:
		opts, args = getopt.getopt(argv,options,longOptions)

	except getopt.GetoptError:
		print('download_segment_videos.py -l <linksFile> -s <segmentsFile> -d <downloadPathOriginalVideos> -p <segmentPathDownload>')
		sys.exit(2)

		
	for opt, arg in opts:

		if opt in ("-h","--help"):
			print('download_segment_videos.py -l <linksFile> -s <segmentsFile> -d <downloadPathOriginalVideos> -p <segmentPathDownload>')
			sys.exit(2)
		elif opt in ("-l","--links"):
			link_file = arg
		elif opt in ("-s","--segment_data"):
			segment_file = arg
		elif opt in ("-d","--downVideos"):
			download_path = arg
		elif opt in ("-p","--downSegments"):
			segments_path = arg

	#Download videos
	download_videos(link_file,segment_file,download_path,segments_path);

if __name__ == "__main__":

	main(sys.argv[1:]);

	
