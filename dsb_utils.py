import os, sys, re, dicom, scipy, cv2
import numpy as np
from skimage import transform, exposure
from sklearn import decomposition

import theano
import theano.tensor as T
import lasagne as nn

#reload(heart)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def volume(x,y):
    d = min(8, np.median(np.diff(x)));
    idx = y>0;
    x = x[idx];
    y = y[idx];
    L = np.sum(idx);
    if L<3:
        return np.nan;
    vol = (y[0]+y[-1])/2.0*d;#end slice
    for i in xrange(L-1):
        vol += (y[i]+y[i+1])*np.abs(x[i+1]-x[i])/2.0;
    return vol/1000.0;


#sorenson-dice
def sorenson_dice(pred, tgt, ss=10):
    return -2*(T.sum(pred*tgt)+ss)/(T.sum(pred) + T.sum(tgt) + ss)

# get_patches deals in 2d arrays of value [0,1]
def get_patches(segment_arr):
    ret = []
    im = segment_arr.astype(np.uint8)
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(cont) for cont in contours[1]] #seems my version of CV2 (3.0) uses [1]
    for contour_idx in xrange(len(hulls)):
        cimg = np.zeros_like(im)
        cv2.drawContours(cimg, hulls, contour_idx, color=255, thickness=-1)
        pts = np.array(np.where(cimg == 255)).T
        ret.append(pts)
    return ret

def ll_of_count(counts, means, stds):
    cm = np.copy(counts)
    cm = (cm*255./cm.max()).astype(np.uint8)
    cm = cm[np.where(cm.sum(axis=1))]
    if cm.shape[0] == 0:
        cm = np.zeros((10, 30), dtype = np.uint8)
    im = Image.fromarray(cm).resize((30,10), Image.ANTIALIAS)
    counts_resized_arr = np.array(im.getdata(), dtype=np.float32).reshape(10,30)/255.
    max_ll = -10000000
    for roll_by in xrange(30):
        resized_counts = np.roll(counts_resized_arr, roll_by, axis=1).flatten()
        ll = 0.
        for i in xrange(resized_counts.shape[0]):
            ll += np.log(scipy.stats.norm.pdf(resized_counts[i], loc=means[i], scale=stds[i]))
        if ll > max_ll:
            max_ll = ll
    return max_ll

def clean_segmentation(segments, img_size):
    mean = segments.mean(axis=(0,1))
    gaussian_params = gaussian2d.moments_fake(mean, normalize_height=True)
    #gaussian_params = gaussian2d.fitgaussian(mean)
    pdf = gaussian2d.gaussian(*gaussian_params)
    seg_binary = np.zeros_like(segments)
    pdf_dict = np.zeros_like(mean)
    for x in xrange(mean.shape[0]):
        for y in xrange(mean.shape[1]):
            pdf_dict[x,y] = pdf(x,y)
    for i in xrange(segments.shape[0]):
        _,sb = cv2.threshold(np.copy(segments[i,0])*255, 127, 255, cv2.THRESH_BINARY)
        patches = get_patches(sb)
        if len(patches)==0:
            continue
        sum_pdf_vals = [sum(pdf_dict[x,y] for x,y in p) for p in patches]
        avg_pdf_vals = [sum(pdf_dict[x,y] for x,y in p)/p.shape[0] for p in patches]
        max_sum_pdf = max(sum_pdf_vals)
        for p_idx, p in enumerate(patches):
            if avg_pdf_vals[p_idx] < 0.07 or sum_pdf_vals[p_idx] < max_sum_pdf:
                for x,y in p:
                    seg_binary[i,0,x,y]=0
            else:
                for x,y in p:
                    seg_binary[i,0,x,y]=1
    return seg_binary

def clean_segmentation2(segments, img_size):
    seg_binary = np.zeros_like(segments)
    for i in xrange(segments.shape[0]):
        _,sb = cv2.threshold(np.copy(segments[i,0])*255, 127, 255, cv2.THRESH_BINARY)
        patches = get_patches(sb)
        if len(patches)==0:
            continue
        sum_pdf_vals = [sum(pdf_dict[x,y] for x,y in p) for p in patches]
        avg_pdf_vals = [sum(pdf_dict[x,y] for x,y in p)/p.shape[0] for p in patches]
        max_sum_pdf = max(sum_pdf_vals)
        for p_idx, p in enumerate(patches):
            if avg_pdf_vals[p_idx] < 0.07 or sum_pdf_vals[p_idx] < max_sum_pdf:
                for x,y in p:
                    seg_binary[i,0,x,y]=0
            else:
                for x,y in p:
                    seg_binary[i,0,x,y]=1
    return seg_binary

def get_contour_shape(x,y,z):
    N = 30;
    res = np.zeros(N);
    cx,cy = np.mean(x),np.mean(y);
    L = x.size;
    theta = (np.arctan2(y-cy,x-cx)*180/np.pi+180+90)%360;#0-360
    b = np.array(np.floor(theta/(360.0001/N)),dtype=np.int);
    for i in range(N):
        idx = (b==i);
        if sum(idx)==0:##bad contour
            return None;
        res[i] = np.mean(z[b==i]);
    return res;

def get_eff_portion(con_shape, cut):
    return np.sum(con_shape<cut)*1.0/con_shape.size;

def get_contour_portion(images,segb):
    ns = images.shape[0];
    nt = images.shape[1];
    portion = np.zeros((ns,nt));
    for s in range(ns):
        for t in range(nt):
            img = images[s,t,0];
            seg = segb[nt*s+t,0];
            if np.sum(seg)<10:
                portion[s,t] = 0.0;
                continue;
            mask = cv2.dilate(seg,np.ones((7,7)))-seg>0;
            z = img[mask];
            x,y = np.where(mask);
            lvinside = np.mean(img[seg>0]);
            lvoutside = np.percentile(z,20);
            ccut = lvinside * 0.3 + lvoutside * 0.7;
            cnt_sh = get_contour_shape(x,y,z);
            if cnt_sh is None:
                portion[s,t] = 0.0;
            else:
                res = get_eff_portion(cnt_sh,ccut);
                portion[s,t] = res;
    return portion;


def write_outputs(dsets, dest_dir,vvv,style):
    areas_lines = [] #the area
    #calc_lines = []
    p_lines = [];

    for dset in dsets:
        areas_lines.append('{},{},{},'.format(dset.name, len(dset.slices_ver),len(dset.time)) +
                ','.join(['%.3f'%(c_) for c_ in dset.slocation]) + ',' +
                ','.join(['%.1f'%(c_) for c_ in dset.areas.T.flatten()]) + '\n')
        p_lines.append('{},{},{},'.format(dset.name, len(dset.slices_ver),len(dset.time)) +
                ','.join(['%.3f'%(c_) for c_ in dset.slocation]) + ',' +
                ','.join(['%.3f'%(c_) for c_ in dset.contour_portion.T.flatten()]) + '\n')
    open(os.path.join(dest_dir, 'areas_map_{}.csv'.format(vvv)), style) \
            .writelines(areas_lines)
    open(os.path.join(dest_dir, 'contour_portion_{}.csv'.format(vvv)), style) \
            .writelines(p_lines)

def clean_counts(counts):
    times_totals = counts.mean(axis=0)
    sys_time, dias_time = np.argmin(times_totals), np.argmax(times_totals)

    ret = np.copy(counts)
    for s in xrange(counts.shape[0]):
        last_t = t = dias_time
        while (t != sys_time):
            t -= 1
            if t == -1:
                t = counts.shape[1]-1
            ret[s,t] = min(ret[s,t], ret[s, last_t])
            last_t = t
        last_t = t = dias_time
        while (t != sys_time):
            t += 1
            if t == counts.shape[1]:
                t = 0
            ret[s,t] = min(ret[s,t], ret[s, last_t])
            last_t = t
    return ret


# calc_map = { dset_name: ([sys_vector], [dias_vector]) }
# vector is of format [1, calculated_val, (any other variables)]
# e.g. { 1: [1, sys_val, variation, ... ], [1, dias_val, variation, ... ]}
# calculates optimal w (four functions) as linear combination of everything
# in the vector
def optimize_w(calc_vector_map, label_map, dims_to_use = -1, function=sigmoid,
        min_w = 1, max_w = 13):
    # slice to fewer dims if specified
    calculated_map = { k:(tuple([v1[:dims_to_use] for v1 in v]) if dims_to_use > 0 else v)
                      for k,v in calc_vector_map.iteritems() if k in label_map }
    lin_constr = lambda x_vec, p_vec: min(max_w, max(min_w, np.dot(x_vec, p_vec)))
    error_funcs = [lambda a: np.concatenate([calculate_diffs(calc[0][1], label_map[ds][1],
                                lin_constr(calc[0], a), 9, function=function)
                            for ds,calc in calculated_map.iteritems()]),
                  lambda a: np.concatenate([calculate_diffs(calc[0][1], label_map[ds][1], 9,
                                lin_constr(calc[0], a), function=function)
                            for ds,calc in calculated_map.iteritems()]),
                  lambda a: np.concatenate([calculate_diffs(calc[1][1], label_map[ds][0],
                                lin_constr(calc[1], a), 9, function=function)
                            for ds,calc in calculated_map.iteritems()]),
                  lambda a: np.concatenate([calculate_diffs(calc[1][1], label_map[ds][0], 9,
                                lin_constr(calc[1], a), function=function)
                            for ds,calc in calculated_map.iteritems()])]
    num_vars = len(calculated_map.values()[0][0])
    guesses = [[5,0.1] + [.01]*(num_vars-2)]*4
    parms = []
    for func, guess in zip(error_funcs, guesses):
        obj, success = scipy.optimize.leastsq(func, guess)
        parms.append(obj)
        print obj
    return lambda p, idx: lin_constr(p, parms[idx])

def calculate_submission_values(volume, width_below, width_above, function=sigmoid):
    ret = []
    for i in xrange(600):
        term = function((i-volume)/(width_below if i < volume else width_above))
        ret.append(term)
    return np.array(ret)

def calculate_diffs(calculated, real, width_below, width_above, function=sigmoid):
    calc_vals = calculate_submission_values(calculated, width_below, width_above, function)
    signals = np.array([1 if i > real else 0 for i in range(600)])
    return signals - calc_vals

def calculate_err(calculated, real, width_below, width_above, function=sigmoid):
    diffs = calculate_diffs(calculated, real, width_below, width_above, function)
    return np.square(diffs).mean()

def get_label_map(labels_file):
    labels = np.loadtxt(labels_file, delimiter=',', skiprows=1)
    label_map = {}
    for l in labels:
        label_map[l[0]] = (l[2], l[1])
    return label_map

def get_calc_counts_errors_maps(calc_file, counts_file, labels_file):
    label_map = get_label_map(labels_file)
    calc_map = read_csv(calc_file, header=None)
    calc_map = dict((r[0], (r[1],r[2])) for _,r in calc_map.iterrows())
    counts_map = None
    if counts_file is not None:
        counts_map = open(counts_file, 'r').readlines()
        counts_map = [l.split(',') for l in counts_map]
        counts_map = [[int(st) for st in l] for l in counts_map]
        counts_map = dict((r[0], np.array(r[2:]).reshape((-1,r[1]))) for r in counts_map)
    def error(calc):
        return 0.5*(calculate_err(calc[0], label_map[ds][1], 10, 10) \
                + calculate_err(calc[1], label_map[ds][0], 10, 10))
    errors_map = dict([(ds,error(calc)) for ds,calc in calc_map.iteritems()
        if ds in label_map])
    return calc_map, counts_map, errors_map

def crop_resize(img, size):
    """crop center and resize"""
    img = img.astype(float) / np.max(img)
    if img.shape[0] < img.shape[1]:
        img = img.T[::-1]
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 64, 64
    resized_img = transform.resize(crop_img, (size, size))
    resized_img *= 255
    return resized_img.astype("uint8")

def rescale(img, sc):
    import pdb; pdb.set_trace()
    res = np.zeros_like(img);
    size = res.shape;
    ns = (int(size[0]*sc),int(size[1]*sc));
    if sc>1:
        sx,sy = (ns[0]-size[0])//2,(ns[1]-size[1])//2;
        res = cv2.resize(img,ns)[sx:sx+size[0],sy:sy+size[1]];
    else:
        sx,sy = (size[0]-ns[0])//2,(size[1]-ns[1])//2;
        res[sx:sx+ns[0],sy:sy+ns[1]] = cv2.resize(img,ns);
    return res;

def img_shift(img, xy):
    res = np.zeros_like(img);
    non = lambda s: s if s<0 else None
    mom = lambda s: max(0,s)
    ox,oy = xy;
    res[mom(oy):non(oy), mom(ox):non(ox)] = img[mom(-oy):non(-oy), mom(-ox):non(-ox)]
    return res;


def segmenter_data_transform(imb, shift=0, rotate=0, scale=0, normalize_pctwise=(20,95), istest=False):
    if isinstance(imb, tuple) and len(imb) == 2:
        imgs,labels = imb
    else:
        imgs = imb

    # rotate image if training
    if rotate>0:
        for i in xrange(imgs.shape[0]):
            degrees = rotate if istest else np.clip(np.random.normal(),-2,2)*rotate;
            imgs[i,0] = scipy.misc.imrotate(imgs[i,0], degrees, interp='bilinear')
            if isinstance(imb, tuple):
                labels[i,0] = scipy.misc.imrotate(labels[i,0], degrees, interp='bilinear')
    #rescale
    """
    if scale>0:
        assert(scale>0 and scale<=0.5);
        for i in xrange(imgs.shape[0]):
            sc = 1 + (scale if istest else np.clip(np.random.normal(),-2,2)*scale);
            imgs[i,0] = rescale(imgs[i,0],sc);
            if isinstance(imb, tuple):
                labels[i,0] = rescale(labels[i,0], sc);
    """
    #shift
    if shift>0 and not istest:
        for i in xrange(imgs.shape[0]):
            x,y = np.random.randint(-shift,shift,2);
            imgs[i,0] = img_shift(imgs[i,0], (x,y));
            if isinstance(imb, tuple):
                labels[i,0] = img_shift(labels[i,0], (x,y));

    imgs = nn.utils.floatX(imgs)/255.0;
    for i in xrange(imgs.shape[0]):
        pclow, pchigh = normalize_pctwise 
        if isinstance(pclow,tuple):
            pclow = np.random.randint(pclow[0],pclow[1]);
            pchigh = np.random.randint(pchigh[0],pchigh[1]);
        pl,ph = np.percentile(imgs[i],(pclow, pchigh))
        imgs[i] = exposure.rescale_intensity(imgs[i], in_range=(pl, ph));
        imgs[i] = 2*imgs[i]/imgs[i].max() - 1.

    if isinstance(imb,tuple):
        #labels = nn.utils.floatX(labels)/255.0;
        return imgs,labels
    else:
        return imgs;

def deconvert(im):
    return ((im-im.min())*255/(im.max()-im.min())).astype(np.uint8)


def z_old_optimize_w(calc_map, label_map):
    calculated_map = dict((k,v) for k,v in calc_map.iteritems() if k in label_map)
    lin_constr = lambda x,a,b: min(20, max(0.5, a*x+b))
    error_funcs = [lambda a: np.concatenate([calculate_diffs(calc[0], label_map[ds][1], lin_constr(calc[0], a[0], a[1]), 9)
                            for ds,calc in calculated_map.iteritems()]),
                  lambda a: np.concatenate([calculate_diffs(calc[0], label_map[ds][1], 9, lin_constr(calc[0], a[0], a[1]))
                            for ds,calc in calculated_map.iteritems()]),
                  lambda a: np.concatenate([calculate_diffs(calc[1], label_map[ds][0], lin_constr(calc[1], a[0], a[1]), 9)
                            for ds,calc in calculated_map.iteritems()]),
                  lambda a: np.concatenate([calculate_diffs(calc[1], label_map[ds][0], 9, lin_constr(calc[1], a[0], a[1]))
                            for ds,calc in calculated_map.iteritems()])]
    guesses = [[0.04656, 4.693],
              [0.03896, -0.4893],
              [0.02458, 1.541],
              [0.03392,0.1355]]
    parms = []
    for func, guess in zip(error_funcs, guesses):
        obj, success = scipy.optimize.leastsq(func, guess)
        parms.append(obj)
    return lambda p, idx: lin_constr(p, parms[idx][0], parms[idx][1])

# given predictions and label map, gives optimal stdev above and below
# for each example
def z_old_optimal_w_funcs(calculated_map, label_map, verbose=False):
    optimal_ws_map = dict((ds,[]) for ds in calculated_map if ds in label_map)
    for ds in [d for d in calculated_map if d in label_map]:
        sys_vol, dias_vol = calculated_map[ds]
        edv, esv = label_map[ds]
        err_func = [lambda x: calculate_err(sys_vol, esv, x, 10),
                   lambda x: calculate_err(sys_vol, esv, 10, x),
                   lambda x: calculate_err(dias_vol, edv, x, 10),
                   lambda x: calculate_err(dias_vol, edv, 10, x)]
        for w_idx in xrange(4):
            min_err = 1000000
            min_w = 0
            for w in xrange(100):
                err = err_func[w_idx](w)
                if err < min_err:
                    min_err = err
                    min_w = w
            optimal_ws_map[ds].append(min_w)
        if verbose and ds % 5 == 0:
            print ds, optimal_ws_map[ds]
    preds_arr = np.empty((len(optimal_ws_map), 6), dtype=np.float32)
    i=0
    for ds in optimal_ws_map:
        preds_arr[i] = np.array([calculated_map[ds][0], calculated_map[ds][1],
                                 min(100,optimal_ws_map[ds][0]),
                                 min(100,optimal_ws_map[ds][1]),
                                 min(100,optimal_ws_map[ds][2]),
                                 min(100,optimal_ws_map[ds][3])])
        i += 1
    degree=1
    wsb1 = np.poly1d(np.polyfit(preds_arr[:,0], preds_arr[:,2], degree))
    wsa1 = np.poly1d(np.polyfit(preds_arr[:,0], preds_arr[:,3], degree))
    wdb1 = np.poly1d(np.polyfit(preds_arr[:,1], preds_arr[:,4], degree))
    wda1 = np.poly1d(np.polyfit(preds_arr[:,1], preds_arr[:,5], degree))
    wsb = lambda x: min(20, max(0, wsb1(x)))
    wsa = lambda x: min(20, max(0, wsa1(x)))
    wdb = lambda x: min(20, max(0, wdb1(x)))
    wda = lambda x: min(20, max(0, wda1(x)))
    return wsb, wsa, wdb, wda

def save_imgcon(cst,img,con=None):#cst (case, slice, time)
    if con is None:
        con = np.zeros_like(img);
    con *= 255;
    import os
    ddir = c.data_auto_contours + '/size_{}'.format(c.fcn_img_size);
    if not os.path.isdir(ddir):
        os.mkdir(ddir);
    fname = ddir + '/c_{}_s_{}_t_{}.pkl'.format(cst[0],cst[1],cst[2]);
    import pickle
    with open(fname,'wb') as f:
        pickle.dump((img,con),f);
