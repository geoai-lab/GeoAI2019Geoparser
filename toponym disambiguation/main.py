from geonameshelper import get_toponym_pop
import time
import numpy as np


users_list = ["yingjiehu","jimmyub","yjhugeo","yhuhotmail", "yhutmail"]


def my_split(strings, char):
    temp = strings.split(char)
    return temp[0], temp[4], temp[5]


def main(filename_r, filename_w):
    geotagging_file = open(filename_r, 'r')
    geocoding_file = open(filename_w, 'w')
    saved_toponyms = {}

    for line in geotagging_file:
        if line is '\n':
            geocoding_file.write('\n')
            continue

        toponyms = line.split('||')
        result_line = ""
        for i in range(0, len(toponyms)-1):
            toponym, start, end = my_split(toponyms[i], ",,")
            if toponym.contain("#"):
            #     toponym = toponym[2:]
            #     print(toponym)

            if toponym not in saved_toponyms.keys():
                if len(toponym) <= 1:
                    continue
                result = get_toponym_pop(toponym, users_list)['geonames']
                if len(result) == 0:
                    continue

                toponymName = result[0]["toponymName"]
                lat = result[0]["lat"]
                lng = result[0]["lng"]
                saved_toponyms[toponym] = [toponymName, lat, lng]
                result_line += toponymName + ",," + toponym + ",," + lat + ",," + lng + ",," + start + ",," + end + "||"
                time.sleep(0.5)
            else:
                result = saved_toponyms[toponym]
                toponymName = result[0]
                lat = result[1]
                lng = result[2]
                result_line += toponymName + ",," + toponym + ",," + lat + ",," + lng + ",," + start + ",," + end + "||"

        geocoding_file.write(result_line + '\n')

    geocoding_file.close()
    geotagging_file.close()
    np.save("TR-News.npy", saved_toponyms)
    return tick


if __name__ == "__main__":
    # create instance of config
    filename = "/path/to/geotagging/result/file"
    filename2 = "result/path"
    # load, train, evaluate and interact with model

    c = main(filename, filename2)

    print(c)
