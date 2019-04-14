import csv

def preprocessing(row):
    row = row.lower()
    row = row.replace("<br />","")
    row = row.replace("\"", "")
    row = row.replace("'","")
    return row
imdb = open('movie_data.csv', 'r', encoding='utf-8')
dataset = csv.reader(imdb)

f = open('output.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
counter = 0
line = []
for row in dataset:
    new_row = preprocessing(row[0])
    #line.insert(counter, [counter,row[0],row[1]])
    wr.writerow([counter, new_row, row[1]])
    #line[counter] = [counter,row[0],row[1]]
    counter += 1

f.close()