import csv

#movie_data.csv => encoding = 'UTF8'

def get_dataset(file, encoding_, type_):
    datas = []
    
    f = open(file, 'r', encoding=encoding_)

    lists = csv.reader(f)
    i = 0
    if type_ == 'part':
        for list_ in lists:
            if i == 0:
                pass
            elif i <= 10000:
                datas.append({'review':list_[0], 'sentiment':list_[1]})
            else:
                break
            i += 1
    elif type_ == 'full':
        for list_ in lists:
            if not i == 0:
                datas.append({'review':list_[0], 'sentiment':list_[1]})
            i += 1

    return datas


            
datas = get_dataset('movie_data.csv', 'UTF8', 'full')
