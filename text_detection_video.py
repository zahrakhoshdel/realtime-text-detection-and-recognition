# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import imutils
import time
import cv2
import serial

ser = serial.Serial('COM4', 9600, timeout=.1)
# از کتابخانه تسراکت براي شناسايي حروف متن استفاده مي شود
pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'

#تابع زير براي هر فريم استفاده مي شود و دو خروجي دارد
#1. مختصات کادر محدوده یک متن
#2.احتمال تشخیص منطقه متن
def decode_predictions(scores, geometry):
        #مختصات و امتياز به دست آمده را دريافت مي کنيم
        # تعداد رديف ها و ستون ها را از امتياز ميگيرد
	(numRows, numCols) = scores.shape[2:4]
	# تعريف متغير براي ذخيره مختصات مناطق متن
	rects = []
	# تعريف متغير براي ذخيره احتمال مربوط به هر منطقه متن
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# امتيازات يا همان احتمالات را استخراج و سپس داده هاي هندسي مورد استفاده
		# براي به دست آوردن مختصات کادر محدود کننده منطقه متن را از آن دريافت ميکنيم
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# حلقه پيمايش ستون هاي هر رديف
		for x in range(0, numCols):
			# مناطقي که حداقل احتمال را براي وجود متن ندارند بررسي نميکنيم
			# با اين کار تشخيص هاي متن ضعيف را حذف ميکنيم
			if scoresData[x] < args["min_confidence"]:
				continue

			# با استفاده از مدل تشخيص دهنده متن، تصوير ورودي 4برابر کوچکتر از تصوير اصلي مي شود (east)
			# بنابراين ما خروجي را 4 برابر بزرگ ميکنيم تا به اندازه اصلي برگردد براي محاسبه مختصات
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# زاويه چرخش براي پيش بيني استخراج مي شود
			# سپس با استفاده از اين زاويه مقدار سينوس و کسينوس محاسبه مي شود
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

                        #مختصات کادر محدوده منطقه متن استخراج مي شود
			# براي به دست آوردن عرض و ارتفاع کادر محدود کننده متن از داده هاي هندسي به دست آمده استفاده ميکنيم
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# نقاط شروع و پايان کادر منطقه متن محاسبه مي شود
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# مختصات کادر محدوده منطقه متن و احتمال آن به روز يا اپديت مي شود
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# تنظيمات پيش فرض ارگومان ها
ap = argparse.ArgumentParser()
# مسير فيلم ورودي. اگر اين مسير وارد شود از دوربين وب کم استفاده نمي شود
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
# آستانه احتمال براي تشخيص متن که حداقل آن به صورت پيش فرض 0.5 در نظر ميگيريم 
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
# تغيير سايز عرض تصوير به صورت پيش فرض 320
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
# تغيير سايز ارتفاع تصوير ، به صورت پيش فرض 320
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# ابعاد اصلي فريم، ابعاد جديد فريم و نسبت بين ابعاد مقدار دهي اوليه مي شوند
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# براي تشخيص متن با استفاده از مدل يادگيري عميق استفاده شده، به خروجي ويژگي هاي دو لايه از شبکه عصبي نياز داريم
# اولي: با استفاده از تابع فعال ساز سيگموئيد، احتمال منطقه متن را به دست مي اوريم
# دومي: با استفاده از داده هاي هندسي به دست آمده از اين لايه ميتوان براي استخراج مختصات کادر محدود کننده متن استفاده کرد
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

#مسير فايل مدل اشکارساز متن 
args1 = {"east":"frozen_east_text_detection.pb"}
# load the pre-trained EAST text detector
# بارگذاري مدل تشخيص دهنده متن از قبل آموزش ديده
print("[INFO] loading EAST text detector...")
#  و استفاده از داده هاي مدل بارگذاري شده شبکه عصبي در حافظه 
net = cv2.dnn.readNet(args1["east"])

# اگر ویدیو از قبل ضبط شده برای تشخیص متن داشتیم می توانیم ان را اجرا کنیم یا در غیر اینصورت وب کم فعال می شود.
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# اگر فايل ويديويي براي تشخيص وجود داشت ويديو اجرا ميشود
else:
	vs = cv2.VideoCapture(args["video"])

# شمارنده فريم بر ثانيه شروع به کار ميکند
fps = FPS().start()
results = []

# حلقه عمليات براي هر فريم در ويديو
while True:
	# فريم فعلي را بررسي ميکنيم
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# ايا اخرين فريم است يا خير
	if frame is None:
		break

	# تغيير اندازه ابعاد فريم 
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# اگر ابعاد فريم none باشد
	# باز هم نياز است ابعاد فريم قديمي به فريم جديد محاسبه شود
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# سپس مجددا فريم را تغيير اندازه مي دهيم (بايد مضرب 32 باشد)
	frame = cv2.resize(frame, (newW, newH))

	# خروجي دو لايه انتخاب شده از شبکه را به دست مي اوريم
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# مختصات و احتمالات به دست آمده از شبکه را به تابعي که در ابتدا تعريف کرده بوديم ميدهيم
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	PlateNumber = ""

	# حلقه براي هر کادر به دست آمده
	for (startX, startY, endX, endY) in boxes:
		# مختصات کادر محدوه منطقه متن را براساس نسبت هاي مربوطه به دست مي اورد
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# کادر محدوده متن را با رنگ سبز دور متن مربوطه ترسيم ميکند
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		roi = orig[startY:endY, startX:endX]

                # تنظيمات مربوط به تشخيص متن انگليسي در فريم با استفاده از کتابخانه تسراکت 
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)  # lstm
		PlateNumber = text+PlateNumber
		

		results.append(((startX, startY, endX, endY), text))
		print("{}\n".format(PlateNumber))
		##text += "\r\n"
		# ارسال متن تشخيص داده شده به پورت سريال
		ser.write(PlateNumber.encode())

	# با خروج از حلقه مقدار فريم بر ثانيه اپديت مي شود
	fps.update()

	# نمايش خروجي در هر فريم
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	# براي خروج دستي از حلقه مي توانيم کليد کیو انگلیسی را فشار دهيم
	if key == ord("q"):
		break

# توقف زمان و نشان دادن اطلاعات فريم بر ثانيه در خروجي
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# اگر از وب کم استفاده کرديم ان را پس از اتمام کار آزاد ميکنيم
if not args.get("video", False):
	vs.stop()

# در غير اينصورت ساير منابع استفاده شده را ازاد ميکنيم
else:
	vs.release()

# بستن پورت سريال
ser.close()
# close all windows
cv2.destroyAllWindows()
