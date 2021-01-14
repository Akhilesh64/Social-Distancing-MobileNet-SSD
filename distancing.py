import cv2
import itertools

def distancing(people_coords, img, dist_thres_lim=(200,250)):
    # Plot lines connecting people
    already_red = dict() # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers = []
    high_risk = []
    for i in people_coords:
        centers.append(((int(i[3])+int(i[1]))//2,(int(i[2])+int(i[0]))//2))
    for j in centers:
        already_red[j] = 0
    x_combs = list(itertools.combinations(people_coords,2))
    radius = 5
    thickness = 2
    for x in x_combs:
        xyxy1, xyxy2 = x[0],x[1]
        cntr1 = ((int(xyxy1[3])+int(xyxy1[1]))//2,(int(xyxy1[2])+int(xyxy1[0]))//2)
        cntr2 = ((int(xyxy2[3])+int(xyxy2[1]))//2,(int(xyxy2[2])+int(xyxy2[0]))//2)
        dist = ((cntr2[0]-cntr1[0])**2 + (cntr2[1]-cntr1[1])**2)**0.5

        if dist > dist_thres_lim[0] and dist < dist_thres_lim[1]:
            color = (0, 255, 255)
            label = "Low Risk "
            cv2.line(img, cntr1, cntr2, color, thickness)
            if already_red[cntr1] == 0:
                cv2.circle(img, cntr1, radius, color, -1)
            if already_red[cntr2] == 0:
                cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                cntr = ((int(xy[3])+int(xy[1]))//2,(int(xy[2])+int(xy[0]))//2)
                if already_red[cntr] == 0:
                    c1, c2 = (int(xy[1]), int(xy[0])), (int(xy[3]), int(xy[2]))
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        elif dist < dist_thres_lim[0]:
            color = (0, 0, 255)
            label = "High Risk"
            high_risk.append(cntr1)
            high_risk.append(cntr2)
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            # Plots one bounding box on image img
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            for xy in x:
                c1, c2 = (int(xy[1]), int(xy[0])), (int(xy[3]), int(xy[2]))
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img, len(set(high_risk))
