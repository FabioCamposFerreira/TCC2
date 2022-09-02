import os


def firt_row(csv_name, methods):
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


def read_cell(csv_name, row_pos, column_pos):
    """Return cell content in csv file

    Parameters
    ----------
    csv_name : str
        ??
    row_pos : int
        ??
    column_pos : int
        ??

    Returns
    -------
    str
        ??
    """
    with open(csv_name, "r") as csv_file:
        for r in range(row_pos):
            columns = csv_file.readline()
        columns = csv_file.readline().split(";")
    return columns[column_pos].strip()


def make_sum(csv_name, row_last, colum_index, rows_len, colums):
    """Create last line (sum line) in csv

    Args:
        csv_name: string

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
    if colums[colum_index] == "Acertou?" or colums[colum_index] == "Tempo de classificação":
        sum = 0
        for row_pos in range(rows_len):
            sum += float(read_cell(csv_name, row_pos+2, colum_index).replace(" segundos",""))
        per = sum/rows_len
        row_last += ";"+str(per)
    elif colum_index == 0:
        row_last += "% de acertos"
    else:
        row_last += ";-"
    return row_last, colum_index


def predict_line(result):
    """??

    Parameters
    ----------
    result : list


    Returns
    -------
    str
        ??
    """
    predict_class = result[0]
    delta_time = result[1]
    correct_class = result[2]
    arq = result[3]
    is_correct = str(int(str(correct_class) == str(predict_class)))
    return arq + ";" + str(correct_class)\
               + ";" + str(predict_class)\
               + ";" + is_correct\
               + ";" + str(delta_time) + " segundos"


def write_row(csv_name, row):
    """Add row in .csv
    Args:
        csv_name : string
            Path of the csv file

    Returns:
        None
    """
    with open(csv_name, "a") as csv_file:
        csv_file.write("\n"+row)


def construct_last_row(csv_name):
    """Create last line (sum line) in csv

    Parameters
    ----------
    csv_name : string
            Path of the csv file

    Returns
    -------
    str
        ??
    """
    with open(csv_name, "r") as csv_file:
        csv_file.readline()
        colums = csv_file.readline().strip().split(";")
        rows_len = len(csv_file.read().split("\n"))
    row_last = ""
    for colum_index in range(len(colums)):
        row_last, colum_index = make_sum(csv_name,
                                         row_last,
                                         colum_index,
                                         rows_len,
                                         colums)
    return row_last


def construct(csv_name, methods, results):
    """??

    Parameters
    ----------
    csv_name : str
        ??
    methods : dic
        ??
    results : list
        ??
    """
    firt_row(csv_name, methods)
    for result in results:
        row = predict_line(result)
        write_row(csv_name, row)
    row = construct_last_row(csv_name)
    write_row(csv_name, row)
