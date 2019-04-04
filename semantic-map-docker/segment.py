import lightnet


model = lightnet.load('yolo')


def find_bboxes(rgb_image, out, thr=0.1):
    img = lightnet.Image.from_bytes(open(out, 'rb').read())

    # Object detection threshold defaults to 0.1 here
    # img = lightnet.Image(rgb_image.astype(np.float32)) #DOCUMENTED BUT ACTUALLY MESSES UP A LOT!!!
    boxes = model(img, thresh=thr)

    #print(boxes)
    # Coordinates in YOLO are relative to center coordinates
    boxs_coord = [(int(x), int(y), int(w), int(h)) for cat, name, conf, (x, y, w, h) in boxes]

    #print(boxs_coord)
    return boxs_coord


def convert_bboxes(box_list, shape, resolution=(416, 312)):

    ow, oh, _ = shape
    tgt_w, tgt_h = resolution

    new_bx = []

    for x, y, w, h in box_list:

        print("Original: (%s, %s, %s, %s)" % (x, y, w, h))

        # Make them absolute from relative
        x_ = x  # *tgt_w
        y_ = y  # *tgt_h
        w_ = w  # *tgt_w
        h_ = h  # *tgt_h

        print("Scaled: (%s, %s, %s, %s)" % (x_, y_, w_, h_))
        # And change coord system for later cropping
        x1 = (x_ - w_ / 2)  # /ow
        y1 = (y_ - h_ / 2)  # /oh
        x2 = (x_ + w_ / 2)  # /ow
        y2 = (y_ + h_ / 2)  # /oh

        # Add check taken from draw_detections method in Darknet's image.c
        if x1 < 0:
            x1 = 0
        if x2 > ow - 1:
            x2 = ow - 1

        if y1 < 0:
            y1 = 0

        if y2 > oh - 1:
            y2 = oh - 1

        print("For ROI: (%s, %s, %s, %s)" % (x1, y1, x2, y2))
        new_bx.append((x1, y1, x2, y2))


    return new_bx


def crop_img(rgb_image, boxs_coord):

    return [ rgb_image[y1:y2, x1:x2] for (x1,y1,x2,y2) in boxs_coord]