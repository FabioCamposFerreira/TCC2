import os


def firts_rows(csv_name, methods):
    """Create the firts lines in csv file
    Args:
        row_last : string
            Path of the image
        colum_index : string
            ??
        rows_len : int
            ??
        classes : list 
            ??

    Returns:
        None
    """
    with open(csv_name, "w") as csv_file:
        row_1 = "Imagem;Classe Correta"
        row_2 = "-;-"
        for method in methods:
            row_1 += ";" + method + ";"+method
            row_2 += ";" + "Rotulo Gerado;Acertou?"
        row_1 += ";"+method
        row_2 += ";"+"Tempo de classificação"
        csv_file.write(row_1)
        csv_file.write("\n"+row_2)


def make_sum(row_last, colum_index, rows_len):
    """Create last line (sum line) in csv

    Args:
        row_last : string
            ??
        colum_index : string
            ??
        rows_len : int
            ??

    Returns:
        row_last: string
            ??
        colum_index : int
            ??
    """
    if colum_index >= 25:
        colum_index -= 26
        row_last, colum_index = make_sum(row_last,
                                         rows_len,
                                         colum_index)
    if colums[colum_index] == "Acertou?" or colums[colum_index] == "Tempo de classificação":
        start_sum = chr(colum_index+65) + str(3)
        end_sum = chr(colum_index+65) + str(rows_len+2)
        row_last += ";=SUM("+start_sum+":"+end_sum+")"+"/"+str(rows_len)
    elif colum_index == 0:
        row_last += "% de acertos"
    else:
        row_last += ";-"
    return row_last, colum_index


def predict_line(predct_class, correct_class, delta_time):
    """Construct row with predict results
    Args:
        predct_class : int
            ??
        correct_class : int
            ??
        delta_time : int
            ??


    Returns:
         :  str

    """
    is_correct = str(int(str(correct_class) == str(predct_class)))
    return ";" + str(predct_class)\
        + ";"+is_correct\
        + ";" + str(delta_time) + " segundos"


def write_row(csv_name, row):
    """Add row in .csv
    Args:
        csv_name : string
            Path of the csv file

    Returns:
        None
    """
    with open(csv_name, "r+") as csv_file:
        csv_file.write(csv_file.read()+"\n"+row)


def write_last_row(csv_name):
    """Create last line (sum line) in csv

    Args:
        csv_name : string
            Path of the csv file

    Returns:
        None
    """
    with open(csv_name, "r+") as csv_file:
        colums = csv_file.readline().split(";")
        rows_len, _ = enumerate(csv_file)
        rows_len -= 2
        row_last = ""
        for colum_index in range(len(colums)):
            row_last, colum_index = make_sum(row_last,
                                             colum_index,
                                             rows_len)
        csv_file.write(row_last+"\n")
