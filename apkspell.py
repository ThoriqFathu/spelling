import streamlit as st
from streamlit_option_menu import option_menu

# import Stem
# from mecs import mecs as Stem
import unicodedata
import mysql.connector
import Levenshtein
import re
import numpy as np

# import pandas as pd
from itertools import zip_longest


def hamming_distance(w1, w2):
    return sum(c1 != c2 for c1, c2 in zip_longest(w1, w2))


def damerau_levenshtein_distance(str1, str2):
    # Matriks untuk menyimpan jarak Damerau-Levenshtein
    d = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Inisialisasi baris pertama dan kolom pertama
    for i in range(len(str1) + 1):
        d[i][0] = i
    for j in range(len(str2) + 1):
        d[0][j] = j

    # Mengisi matriks berdasarkan operasi penyisipan, penghapusan, penggantian, dan transposisi
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # Operasi penghapusan
                d[i][j - 1] + 1,  # Operasi penyisipan
                d[i - 1][j - 1] + cost,  # Operasi penggantian
            )

            # Operasi transposisi
            if (
                i > 1
                and j > 1
                and str1[i - 1] == str2[j - 2]
                and str1[i - 2] == str2[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)

    return d[len(str1)][len(str2)]


def jaro_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    if len_str1 == 0 or len_str2 == 0:
        return 0

    # Hitung batas pencarian (mengikuti definisi Jaro)
    search_range = max(len_str1, len_str2) // 2 - 1

    # Array untuk menandai karakter yang sudah cocok
    match_flags_str1 = [False] * len_str1
    match_flags_str2 = [False] * len_str2

    # Hitung karakter yang cocok dan berada dalam jarak tertentu satu sama lain
    match_count = 0
    for i in range(len_str1):
        low = max(0, i - search_range)
        high = min(i + search_range + 1, len_str2)

        for j in range(low, high):
            if not match_flags_str2[j] and str1[i] == str2[j]:
                match_flags_str1[i] = True
                match_flags_str2[j] = True
                match_count += 1
                break

    if match_count == 0:
        return 0

    # Hitung karakter yang cocok tetapi tidak berada dalam urutan yang sama
    transposition_count = 0
    j = 0
    for i in range(len_str1):
        if match_flags_str1[i]:
            while not match_flags_str2[j]:
                j += 1
            if str1[i] != str2[j]:
                transposition_count += 1
            j += 1

    return (
        match_count / len_str1
        + match_count / len_str2
        + (match_count - transposition_count / 2) / match_count
    ) / 3


def jaro_winkler_distance(str1, str2, p=0.1, L=4):
    jaro_score = jaro_distance(str1, str2)

    # Hitung panjang awalan yang cocok
    prefix_length = 0
    for i in range(min(L, min(len(str1), len(str2)))):
        if str1[i] == str2[i]:
            prefix_length += 1
        else:
            break

    # Hitung Jaro-Winkler Similarity
    jaro_winkler_distance = jaro_score + p * prefix_length * (1 - jaro_score)
    return jaro_winkler_distance


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    # database="skripsi"
    database="madureseset",
)


with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Speling Correction", "String Similarity", "About"],
        default_index=0,
    )

if selected == "Speling Correction":
    # Judul halaman
    st.title("Spelling Correction for Madurese")

    tag_hint = """
        <div style="background-color: #fdd271; width: 650px; padding: 10px;">
            <h5>Hint &#x1F4A1;</h5>
            <p>Typing Madurese accented characters:</p>
            <div style="display: flex;">
                <div>
                    <table style="width:600px; text-align:center; margin:auto;">
                    <tr>
                        <th style="border: solid 1px black;">Accented Characters</th>
                        <th style="border: solid 1px black;">Typing Keys</th>
                        <th style="border: solid 1px black;">Example</th>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">â</td>
                        <td style="border: solid 1px black;">^a</td>
                        <td style="border: solid 1px black;">ab^a' &rarr; abâ'</td>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">è</td>
                        <td style="border: solid 1px black;">`e</td>
                        <td style="border: solid 1px black;">l`eker &rarr; lèker</td>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">ḍ</td>
                        <td style="border: solid 1px black;">.d</td>
                        <td style="border: solid 1px black;">a.d.dhep &rarr; aḍḍhep</td>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">ṭ</td>
                        <td style="border: solid 1px black;">.t</td>
                        <td style="border: solid 1px black;">an.tok &rarr; anṭok</td>
                    </tr>
                    </table>
                </div>
            </div>
                
         </div>
            
    """
    st.markdown(tag_hint, unsafe_allow_html=True)

    # Daftar pilihan untuk dropdown
    options = [
        "Choose Method",
        "Hamming Distance",
        "Levenshtein Distance",
        "Damerau-Levenshtein Distance",
        "Jaro Similarity",
        "Jaro-Winkler Similarity",
    ]

    # Menampilkan dropdown
    st.markdown(
        "<p style='margin-bottom: -50px;'><strong>Method:</strong></p>",
        unsafe_allow_html=True,
    )
    selected_option = st.selectbox(" ", options)

    # Menampilkan hasil seleksi
    # st.write("Anda memilih:", selected_option)

    if selected_option != "Choose Method":
        st.markdown(
            "<p style='margin-bottom: -50px;'><strong>Input:</strong></p>",
            unsafe_allow_html=True,
        )
        user_input = st.text_input(
            "\t", placeholder="Enter misspelled (Madurese) string"
        )

        # # Tampilkan hasil input
        # st.write("Anda memasukkan teks:", user_input)

        # # Tambahkan tombol
        if st.button("Check"):
            # st.write(user_input)
            if user_input == "":
                st.warning("Please input string!")
            else:
                pola = re.compile(r"\^a")
                user_input = re.sub(pola, "â", user_input)
                pola = re.compile(r"\`e")
                user_input = re.sub(pola, "è", user_input)
                pola = re.compile(r"\.d")
                user_input = re.sub(pola, "ḍ", user_input)
                pola = re.compile(r"\.t")
                user_input = re.sub(pola, "ṭ", user_input)
                user_input = user_input.lower()
                normalized_char1 = unicodedata.normalize("NFD", user_input)
                lenStr = len(normalized_char1)
                mycursor = mydb.cursor()

                mycursor.execute("SELECT basic_lemma FROM lemmata")

                myresult = mycursor.fetchall()
                mycursor.close()
                dis = {}
                for data in myresult:
                    # print(len(data[0]))
                    normalized_char2 = unicodedata.normalize("NFD", data[0])
                    if selected_option == "Hamming Distance":
                        hamm = hamming_distance(normalized_char1, normalized_char2)
                        dis[normalized_char2] = hamm
                    # hammingDis.append(dis)
                    elif selected_option == "Levenshtein Distance":
                        lev = Levenshtein.distance(normalized_char1, normalized_char2)
                        dis[normalized_char2] = lev

                    elif selected_option == "Damerau-Levenshtein Distance":
                        dld = damerau_levenshtein_distance(
                            normalized_char1, normalized_char2
                        )
                        dis[normalized_char2] = dld
                    elif selected_option == "Jaro Similarity":
                        jar = jaro_distance(normalized_char1, normalized_char2)
                        if jar <= 1:
                            dis[normalized_char2] = jar
                    else:
                        jarw = jaro_winkler_distance(normalized_char1, normalized_char2)
                        dis[normalized_char2] = jarw

                correct = False
                if (
                    selected_option == "Jaro Similarity"
                    or selected_option == "Jaro-Winkler Similarity"
                ):
                    dis = dict(
                        sorted(dis.items(), key=lambda item: item[1], reverse=True)
                    )
                    topKey = list(dis.keys())[0]
                    topValue = dis[topKey]
                    if topValue == 1:
                        correct = True
                else:
                    dis = dict(sorted(dis.items(), key=lambda item: item[1]))
                    topKey = list(dis.keys())[0]
                    topValue = dis[topKey]
                    if topValue == 0:
                        correct = True

                # st.write("Output:")
                st.markdown("<p><strong>Output:</strong></p>", unsafe_allow_html=True)
                if correct == True:
                    st.success("The word is correct!")
                else:
                    count = 0
                    limit = 5
                    html_spell = f"<table style='text-align: center;'><tr><th>Rank</th><th>Correction Suggestion</th><th>{selected_option}</th></tr>"
                    for key, value in dis.items():
                        # st.warning(f"({user_input} , {key}) = {value}")
                        html_spell += f"<tr><td>{count+1}</td><td>{key}</td><td>{round(value,3)}</td></tr>"
                        count += 1
                        if count == limit:
                            break
                    html_spell += "</table>"
                    st.markdown(html_spell, unsafe_allow_html=True)


# function to page similarity


def cal_hamming(w1, w2):
    indx_str = []
    clm_term = []
    len_str1 = len(w1)
    len_str2 = len(w2)
    mat = np.zeros((max(len(w1), len(w2)), max(len(w1), len(w2))))
    # mat[0][0] = 1
    # print(mat)
    # st.write(max(len(w1), len(w2)))
    itr = 0
    result = 0
    for c1, c2 in zip_longest(w1, w2):
        indx_str.append(c1)
        clm_term.append(c2)
        # st.write(clm_term, indx_str)
        if c1 != c2:
            mat[itr][itr] = 1
            result += 1

        itr += 1
    # df = pd.DataFrame(mat, index=indx_str, columns=clm_term)
    # df = pd.DataFrame(mat)
    # st.dataframe(df)
    # st.table(mat)
    html_ = "<table style='text-align:center; border: 2px solid black;'>"
    # st.write(len(mat))
    for i in range(len(mat) + 1):
        html_ += "<tr>"
        for y in range(len(mat) + 1):
            if i == 0 and y == 0:
                html_ += "<th style='border: 2px solid black;'>s\\t</th>"
            elif i == 0:
                html_ += (
                    f"<th style='border-bottom: 2px solid black;'>{clm_term[y-1]}</th>"
                )
            elif y == 0:
                html_ += (
                    f"<th style='border-right: 2px solid black;'>{indx_str[i-1]}</th>"
                )
            # elif i - 1 in truePos and i == y:
            #     html_ += "<td style='background-color:cyan'>1</td>"
            # elif i - 1 in transPos and i == y:
            #     html_ += "<td style='background-color:yellow'>1</td>"
            else:
                if mat[i - 1][y - 1] == 1:
                    html_ += (
                        f"<td style='background-color:cyan'>{int(mat[i-1][y-1])}</td>"
                    )
                else:
                    html_ += f"<td>{int(mat[i-1][y-1])}</td>"

        html_ += "</tr>"
    html_ += "</table>"
    st.success(f'Hamming Distance between "{w1}" and "{w2}" = {result}')
    st.markdown("<p><strong>Detail:</strong></p>", unsafe_allow_html=True)
    st.markdown(
        html_,
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("List of abbreviations:")
    st.markdown(
        "<table style='text-align: center'><tr><th>Notation</th><th>Description</th></tr><tr><td>s</td><td>Source string</td></tr><tr><td>t</td><td>Target string</td></tr></table>",
        unsafe_allow_html=True,
    )


def cal_levenshtein_distance(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    index_str = [""]
    clm_term = [""]
    # Matriks untuk menyimpan jarak Levenshtein
    matrix = [[0 for _ in range(len_str2)] for _ in range(len_str1)]
    # print(len(matrix[0]))

    # Inisialisasi matriks
    for i in range(len_str1):
        matrix[i][0] = i

    for j in range(len_str2):
        matrix[0][j] = j

    # Menghitung jarak Levenshtein
    for i in range(1, len_str1):
        index_str.append(str1[i - 1])
        for j in range(1, len_str2):
            if i == 1:
                clm_term.append(str2[j - 1])
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            val = min(
                matrix[i - 1][j] + 1,  # Deletion
                matrix[i][j - 1] + 1,  # Insertion
                matrix[i - 1][j - 1] + cost,  # Substitution
            )
            matrix[i][j] = val

    # pd.set_option("mode.chained_assignment", "warn")
    # df = pd.DataFrame(matrix, index=index_str, columns=clm_term)
    # df = pd.DataFrame(matrix)
    # df.columns = pd.io.common.duplicates(df.columns)
    result = matrix[len_str1 - 1][len_str2 - 1]
    # st.dataframe(df)
    html_ = "<table style='text-align:center; border: 2px solid black;'>"
    for i in range(len_str1 + 1):
        html_ += "<tr>"
        for y in range(len_str2 + 1):
            if i == 0 and y == 0:
                html_ += "<th style='border: 2px solid black;'>s\\t</th>"
            elif i == 0 and y == 1:
                html_ += "<th style='border-bottom: 2px solid black;'>''</th>"
            elif i == 1 and y == 0:
                html_ += "<th style='border-right: 2px solid black;'>''</th>"
            elif i == 0:
                html_ += f"<th style='border-bottom: 2px solid black;'>{str2[y-2]}</th>"
            elif y == 0:
                html_ += f"<th style='border-right: 2px solid black;'>{str1[i-2]}</th>"
            # elif i - 1 in truePos and i == y:
            #     html_ += "<td style='background-color:cyan'>1</td>"
            # elif i - 1 in transPos and i == y:
            #     html_ += "<td style='background-color:yellow'>1</td>"
            else:
                if i == len_str1 and y == len_str2:
                    html_ += (
                        f"<td style='background-color:cyan'>{matrix[i-1][y-1]}</td>"
                    )
                else:
                    html_ += f"<td>{matrix[i-1][y-1]}</td>"

        html_ += "</tr>"
    html_ += "</table>"
    st.success(f'Levenshtein distance between "{str1}" and "{str2}" = {result}')
    st.markdown("<p><strong>Detail:</strong></p>", unsafe_allow_html=True)
    st.markdown(
        html_,
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("List of abbreviations:")
    st.markdown(
        "<table style='text-align: center'><tr><th>Notation</th><th>Description</th></tr><tr><td>s</td><td>Source string</td></tr><tr><td>t</td><td>Target string</td></tr></table>",
        unsafe_allow_html=True,
    )


def cal_damerau_levenshtein_distance(str1, str2):
    # Matriks untuk menyimpan jarak Damerau-Levenshtein
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    d = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Inisialisasi baris pertama dan kolom pertama
    for i in range(len(str1) + 1):
        d[i][0] = i
    for j in range(len(str2) + 1):
        d[0][j] = j

    # Mengisi matriks berdasarkan operasi penyisipan, penghapusan, penggantian, dan transposisi
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # Operasi penghapusan
                d[i][j - 1] + 1,  # Operasi penyisipan
                d[i - 1][j - 1] + cost,  # Operasi penggantian
            )

            # Operasi transposisi
            if (
                i > 1
                and j > 1
                and str1[i - 1] == str2[j - 2]
                and str1[i - 2] == str2[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)

    # df = pd.DataFrame(d)
    result = d[len(str1)][len(str2)]
    # st.dataframe(df)
    html_ = "<table style='text-align:center;border: 2px solid black;'>"
    for i in range(len_str1 + 1):
        html_ += "<tr>"
        for y in range(len_str2 + 1):
            if i == 0 and y == 0:
                html_ += "<th style='border: 2px solid black;'>s\\t</th>"
            elif i == 0 and y == 1:
                html_ += "<th style='border-bottom: 2px solid black;'>''</th>"
            elif i == 1 and y == 0:
                html_ += "<th style='border-right: 2px solid black;'>''</th>"
            elif i == 0:
                html_ += f"<th style='border-bottom: 2px solid black;'>{str2[y-2]}</th>"
            elif y == 0:
                html_ += f"<th style='border-right: 2px solid black;'>{str1[i-2]}</th>"
            # elif i - 1 in truePos and i == y:
            #     html_ += "<td style='background-color:cyan'>1</td>"
            # elif i - 1 in transPos and i == y:
            #     html_ += "<td style='background-color:yellow'>1</td>"
            else:
                if i == len_str1 and y == len_str2:
                    html_ += f"<td style='background-color:cyan'>{d[i-1][y-1]}</td>"
                else:
                    html_ += f"<td>{d[i-1][y-1]}</td>"

        html_ += "</tr>"
    html_ += "</table>"
    st.success(f'Damerau-Levenshtein distance between "{str1}" and "{str2}" = {result}')
    st.markdown("<p><strong>Detail:</strong></p>", unsafe_allow_html=True)
    st.markdown(
        html_,
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("List of abbreviations:")
    st.markdown(
        "<table style='text-align: center'><tr><th>Notation</th><th>Description</th></tr><tr><td>s</td><td>Source string</td></tr><tr><td>t</td><td>Target string</td></tr></table>",
        unsafe_allow_html=True,
    )


def cal_jaro_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    if len_str1 == 0 or len_str2 == 0:
        return 0

    # Hitung batas pencarian (mengikuti definisi Jaro)
    search_range = max(len_str1, len_str2) // 2 - 1

    # Array untuk menandai karakter yang sudah cocok
    match_flags_str1 = [False] * len_str1
    match_flags_str2 = [False] * len_str2

    truePos = []
    transPos = []

    # Hitung karakter yang cocok dan berada dalam jarak tertentu satu sama lain
    match_count = 0
    for i in range(len_str1):
        low = max(0, i - search_range)
        high = min(i + search_range + 1, len_str2)
        # st.write(f"low = {low}")
        # st.write(f"high = {high}")
        for j in range(low, high):
            if not match_flags_str2[j] and str1[i] == str2[j]:
                # if i == j:
                #     truePos.append(i)
                # else:
                #     transPos.append(i)
                # st.write(i, j)
                # st.write(str1[i], str2[j])
                match_flags_str1[i] = True
                match_flags_str2[j] = True
                # st.write(i)
                truePos.append(i)
                match_count += 1
                break

    # if match_count == 0:
    #     return 0

    # Hitung karakter yang cocok tetapi tidak berada dalam urutan yang sama
    transposition_count = 0
    j = 0
    for i in range(len_str1):
        if match_flags_str1[i]:
            while not match_flags_str2[j]:
                j += 1
            # st.write(j)
            if str1[i] != str2[j]:
                # st.write(i)
                transposition_count += 1
                transPos.append(i)
                # st.write(str1[i], str2[j])
                # st.write(match_flags_str1[i], match_flags_str2[j])
            j += 1
    # st.write(match_flags_str1, match_flags_str2)

    if match_count == 0:
        return 0, match_count, transposition_count, truePos, transPos
    else:
        result = (
            match_count / len_str1
            + match_count / len_str2
            + (match_count - transposition_count / 2) / match_count
        ) / 3
        # st.write(
        #     f"JD('{str1},{str2}') = 1/3 ( ({match_count} / {len_str1}) + ({match_count} / {len_str2}) + ( ({match_count} - {transposition_count/2})/{match_count}) ) = {result}"
        # )
        return result, match_count, transposition_count, truePos, transPos


def cal_jaro_winkler_distance(str1, str2, p=0.1, L=4):
    jaro_score, match_count, transposition_count, truePos, transPos = cal_jaro_distance(
        str1, str2
    )

    # Hitung panjang awalan yang cocok
    prefix_length = 0
    index_prefix = []
    for i in range(min(L, min(len(str1), len(str2)))):
        if str1[i] == str2[i]:
            # st.write(i)
            # st.write(str1[i], str2[i])
            prefix_length += 1
            index_prefix.append(i)
        else:
            break

    # Hitung Jaro-Winkler Similarity
    jaro_winkler_distance = jaro_score + p * prefix_length * (1 - jaro_score)

    return (
        jaro_winkler_distance,
        jaro_score,
        match_count,
        transposition_count,
        truePos,
        transPos,
        prefix_length,
        p,
        index_prefix,
    )


if selected == "String Similarity":
    st.title("String Similarity")

    tag_hint = """
        <div style="background-color: #fdd271; width: 650px; padding: 10px;">
            <h5>Hint &#x1F4A1;</h5>
            <p>Typing Madurese accented characters:</p>
            <div style="display: flex;">
                <div>
                    <table style="width:600px; text-align:center; margin:auto;">
                    <tr>
                        <th style="border: solid 1px black;">Accented Characters</th>
                        <th style="border: solid 1px black;">Typing Keys</th>
                        <th style="border: solid 1px black;">Example</th>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">â</td>
                        <td style="border: solid 1px black;">^a</td>
                        <td style="border: solid 1px black;">ab^a' &rarr; abâ'</td>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">è</td>
                        <td style="border: solid 1px black;">`e</td>
                        <td style="border: solid 1px black;">l`eker &rarr; lèker</td>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">ḍ</td>
                        <td style="border: solid 1px black;">.d</td>
                        <td style="border: solid 1px black;">a.d.dhep &rarr; aḍḍhep</td>
                    </tr>
                    <tr>
                        <td style="border: solid 1px black;">ṭ</td>
                        <td style="border: solid 1px black;">.t</td>
                        <td style="border: solid 1px black;">an.tok &rarr; anṭok</td>
                    </tr>
                    </table>
                </div>
            </div>
                
         </div>
            
    """
    st.markdown(tag_hint, unsafe_allow_html=True)
    # Daftar pilihan untuk dropdown
    options = [
        "Choose Method",
        "Hamming Distance",
        "Levenshtein Distance",
        "Damerau-Levenshtein Distance",
        "Jaro Similarity",
        "Jaro-Winkler Similarity",
    ]

    # Menampilkan dropdown
    st.markdown(
        "<p style='margin-bottom: -50px;'><strong>Method:</strong></p>",
        unsafe_allow_html=True,
    )
    selected_option = st.selectbox("", options)
    if selected_option != "Choose Method":
        st.markdown(
            "<p style='margin-bottom: -50px;'><strong>Input Source String:</strong></p>",
            unsafe_allow_html=True,
        )
        term1 = st.text_input("\t", placeholder="Enter source (Madurese) string")
        st.markdown(
            "<p style='margin-bottom: -50px;'><strong>Input Target String:</strong></p>",
            unsafe_allow_html=True,
        )
        term2 = st.text_input("\t\t", placeholder="Enter target (Madurese) string")

        # # Tampilkan hasil input
        # st.write("Anda memasukkan teks:", user_input)

        # # Tambahkan tombol
        if st.button("Calculate"):
            if term1 == "" or term2 == "":
                st.warning("Please input string!")
            else:
                pola = re.compile(r"\^a")
                term1 = re.sub(pola, "â", term1)
                pola = re.compile(r"\`e")
                term1 = re.sub(pola, "è", term1)
                pola = re.compile(r"\.d")
                term1 = re.sub(pola, "ḍ", term1)
                pola = re.compile(r"\.t")
                term1 = re.sub(pola, "ṭ", term1)
                pola = re.compile(r"\^a")
                term2 = re.sub(pola, "â", term2)
                pola = re.compile(r"\`e")
                term2 = re.sub(pola, "è", term2)
                pola = re.compile(r"\.d")
                term2 = re.sub(pola, "ḍ", term2)
                pola = re.compile(r"\.t")
                term2 = re.sub(pola, "ṭ", term2)
                normalized_char1 = unicodedata.normalize("NFD", term1.lower())
                normalized_char2 = unicodedata.normalize("NFD", term2.lower())
                st.markdown("<p><strong>Output:</strong></p>", unsafe_allow_html=True)
                if selected_option == "Hamming Distance":
                    # result = hamming_distance(normalized_char1, normalized_char2)
                    cal_hamming(normalized_char1, normalized_char2)

                # hammingDis.append(dis)
                elif selected_option == "Levenshtein Distance":
                    # result = Levenshtein.distance(normalized_char1, normalized_char2)
                    cal_levenshtein_distance(normalized_char1, normalized_char2)

                elif selected_option == "Damerau-Levenshtein Distance":
                    # result = damerau_levenshtein_distance(
                    #     normalized_char1, normalized_char2
                    # )
                    cal_damerau_levenshtein_distance(normalized_char1, normalized_char2)
                elif selected_option == "Jaro Similarity":
                    len_str1 = len(normalized_char1)
                    len_str2 = len(normalized_char2)
                    # result = jaro_distance(normalized_char1, normalized_char2)
                    (
                        result,
                        match_count,
                        transposition_count,
                        truePos,
                        transPos,
                    ) = cal_jaro_distance(normalized_char1, normalized_char2)
                    if result != 0:
                        st.markdown(
                            f"""<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro Similarity between '{normalized_char1}' and '{normalized_char2}' = <b>{round(result,3)}</b></div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        # st.write(
                        #     f"JD('{normalized_char1},{normalized_char2}') = 1/3 ( ({match_count} / {len_str1}) + ({match_count} / {len_str2}) + ( ({match_count} - {0})/{match_count}) ) = {0}"
                        # )
                        st.markdown(
                            f"<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro Similarity between '{normalized_char1}' and '{normalized_char2}' = <b>{0}</b></div>",
                            unsafe_allow_html=True,
                        )
                        # st.warning("No matching characters!")
                    st.markdown(
                        "<p style='padding-top: 10px;'><strong>Detail:</strong></p>",
                        unsafe_allow_html=True,
                    )
                    html_ = (
                        "<table style='text-align: center;border: 2px solid black;'>"
                    )
                    for i in range(len_str1 + 1):
                        html_ += "<tr>"
                        for y in range(len_str2 + 1):
                            if i == 0 and y == 0:
                                html_ += (
                                    "<th style='border: 2px solid black;'>s\\t</th>"
                                )
                            elif i == 0:
                                html_ += f"<th style='border-bottom: 2px solid black;'>{normalized_char2[y-1]}</th>"
                            elif y == 0:
                                html_ += f"<th style='border-right: 2px solid black;'>{normalized_char1[i-1]}</th>"
                            elif i - 1 in transPos and i == y:
                                html_ += "<td style='background-color:yellow'>1</td>"
                            elif i - 1 in truePos and i == y:
                                html_ += "<td style='background-color:cyan'>1</td>"
                            else:
                                html_ += "<td>0</td>"

                        html_ += "</tr>"
                    html_ += "</table>"
                    st.markdown(
                        html_,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.write("Equation:")
                    if result == 0:
                        st.markdown(
                            """<table style='text-align: center'><tr><th>Method</th><th>Equation</th><th>Condition</th></tr><tr><td rowspan='2'>JD(s,t) =</td><td style='background-color:#d4edda;'>0</td><td style='background-color:#d4edda;'>if m = 0</td></tr><tr><td><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <mfrac>
                                    <mn>1</mn>
                                    <mn>3</mn>
                                </mfrac>
                                <mo>(</mo>
                                <mrow>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|s|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|t|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mrow>
                                            <mi>z</mi>
                                            <mo>-</mo>
                                            <mi>y</mi>
                                        </mrow>
                                        <mi>z</mi>
                                    </mfrac>
                                </mrow>
                                <mo>)</mo>
                            </math></td><td>otherwise</td></tr></table>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """<table style='text-align: center'>
                            <tr><th>Method</th><th>Equation</th><th>Condition</th></tr>
                            <tr><td rowspan='2'>JD(s,t) =</td><td>0</td><td>if m = 0</td></tr>
                            <tr style='background-color:#d4edda;'><td><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <mfrac>
                                    <mn>1</mn>
                                    <mn>3</mn>
                                </mfrac>
                                <mo>(</mo>
                                <mrow>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|s|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|t|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mrow>
                                            <mi>z</mi>
                                            <mo>-</mo>
                                            <mi>y</mi>
                                        </mrow>
                                        <mi>z</mi>
                                    </mfrac>
                                </mrow>
                                <mo>)</mo>
                            </math></td><td>otherwise</td></tr></table>""",
                            unsafe_allow_html=True,
                        )
                    st.write("")
                    st.write("List of abbreviations:")
                    st.markdown(
                        "<table style='text-align: center'><tr><th>Notation</th><th>Description</th></tr><tr><td>z</td><td>The number of matching characters</td></tr><tr><td>y</td><td>The number of transpositions divided by 2</td></tr><tr><td>s</td><td>Source string</td></tr><tr><td>t</td><td>Target string</td></tr><tr><td>|s|</td><td>Number of source string characters</td></tr><tr><td>|t|</td><td>Number of target string characters</td></tr><tr><td>JD(s,t)</td><td>Jaro Similarity function</td></tr></table>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.write("Color description:")
                    st.markdown(
                        "<table style='text-align: center'><tr><th>Color</th><th>Description</th></tr><tr><td style='background-color:cyan'></td><td>The number of matching characters</td></tr><tr><td style='background-color:yellow'></td><td>The number of transpositions</td></tr></table>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.write("Result:")
                    if result != 0:
                        # st.markdown(
                        #     f"<div style='background-color:#d4edda;padding: 20px;color: #155724;'>JD('{normalized_char1},{normalized_char2}') = 1/3 ( ({match_count} / {len_str1}) + ({match_count} / {len_str2}) + ( ({match_count} - {transposition_count/2})/{match_count}) ) = <b>{result}</b></div>",
                        #     unsafe_allow_html=True,
                        # )
                        st.write(f"z = {match_count}")
                        st.write(f"|s| = {len_str1}")
                        st.write(f"|t| = {len_str2}")
                        st.markdown(
                            f"""y = <math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>{transposition_count}</mn><mn>{2}</mn></mfrac>{transposition_count}/2 </math> = {transposition_count/2}""",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"""<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro Similarity between '{normalized_char1}' and '{normalized_char2}' = <math xmlns="http://www.w3.org/1998/Math/MathML">
                                <mfrac>
                                    <mn>1</mn>
                                    <mn>3</mn>
                                </mfrac>
                                <mo>(</mo>
                                <mrow>
                                    <mfrac>
                                        <mi>{match_count}</mi>
                                        <mi>{len_str1}</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mi>{match_count}</mi>
                                        <mi>{len_str2}</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mrow>
                                            <mi>{match_count}</mi>
                                            <mo>-</mo>
                                            <mi>{transposition_count/2}</mi>
                                        </mrow>
                                        <mi>{match_count}</mi>
                                    </mfrac>
                                </mrow>
                                <mo>)</mo>
                            </math> = <b>{round(result,3)}</b></div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        # st.write(
                        #     f"JD('{normalized_char1},{normalized_char2}') = 1/3 ( ({match_count} / {len_str1}) + ({match_count} / {len_str2}) + ( ({match_count} - {0})/{match_count}) ) = {0}"
                        # )
                        st.markdown(
                            f"<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro Similarity between '{normalized_char1}' and '{normalized_char2}' = <b>{0}</b></div>",
                            unsafe_allow_html=True,
                        )
                        st.warning("No matching characters!")

                else:
                    # result = jaro_winkler_distance(normalized_char1, normalized_char2)
                    (
                        jaro_winkler_distance,
                        jaro_score,
                        match_count,
                        transposition_count,
                        truePos,
                        transPos,
                        prefix_length,
                        p,
                        index_prefix,
                    ) = cal_jaro_winkler_distance(normalized_char1, normalized_char2)
                    len_str1 = len(normalized_char1)
                    len_str2 = len(normalized_char2)
                    # result = jaro_distance(normalized_char1, normalized_char2)
                    (
                        result,
                        match_count,
                        transposition_count,
                        truePos,
                        transPos,
                    ) = cal_jaro_distance(normalized_char1, normalized_char2)
                    st.markdown(
                        f"""<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro-Winkler Similarity between '{normalized_char1}' and '{normalized_char2}' = <b>{round(jaro_winkler_distance,3)}</b></div>""",
                        unsafe_allow_html=True,
                    )
                    html_ = "<table style='text-align:center;border: 2px solid black;'>"
                    for i in range(len_str1 + 1):
                        html_ += "<tr>"
                        for y in range(len_str2 + 1):
                            if i == 0 and y == 0:
                                html_ += (
                                    "<th style='border: 2px solid black;'>s\\t</th>"
                                )
                            elif i == 0:
                                html_ += f"<th style='border-bottom: 2px solid black;'>{normalized_char2[y-1]}</th>"
                            elif y == 0:
                                html_ += f"<th style='border-right: 2px solid black;'>{normalized_char1[i-1]}</th>"
                            elif i - 1 in index_prefix and i == y:
                                html_ += "<td style='background-color:orange'>1</td>"
                            elif i - 1 in transPos and i == y:
                                html_ += "<td style='background-color:yellow'>1</td>"
                            elif i - 1 in truePos and i == y:
                                html_ += "<td style='background-color:cyan'>1</td>"
                            else:
                                html_ += "<td>0</td>"

                        html_ += "</tr>"
                    html_ += "</table>"
                    st.markdown(
                        "<p style='padding-top: 10px;'><strong>Detail:</strong></p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        html_,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.write("Equation:")
                    if result == 0:
                        st.markdown(
                            """<table style='text-align: center'><tr><th>Method</th><th>Equation</th><th>Condition</th></tr><tr><td rowspan='2'>JD(s,t) =</td><td style='background-color:#d4edda;'>0</td><td style='background-color:#d4edda;'>if m = 0</td></tr><tr><td><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <mfrac>
                                    <mn>1</mn>
                                    <mn>3</mn>
                                </mfrac>
                                <mo>(</mo>
                                <mrow>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|s|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|t|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mrow>
                                            <mi>z</mi>
                                            <mo>-</mo>
                                            <mi>y</mi>
                                        </mrow>
                                        <mi>z</mi>
                                    </mfrac>
                                </mrow>
                                <mo>)</mo>
                            </math></td><td>otherwise</td></tr></tr><tr><td>JWD(s,t)</td><td style='background-color:#d4edda;'>JD(s,t) + p * q (1 - JD(s,t))</td><td style='background-color:#d4edda;'>All</td></tr></table>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """<table style='text-align: center'><tr><th>Method</th><th>Equation</th><th>Condition</th></tr><tr><td rowspan='2'>JD(s,t) =</td><td>0</td><td>if m = 0</td></tr><tr style='background-color:#d4edda;'><td><math xmlns="http://www.w3.org/1998/Math/MathML">
                                <mfrac>
                                    <mn>1</mn>
                                    <mn>3</mn>
                                </mfrac>
                                <mo>(</mo>
                                <mrow>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|s|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mi>z</mi>
                                        <mi>|t|</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mrow>
                                            <mi>z</mi>
                                            <mo>-</mo>
                                            <mi>y</mi>
                                        </mrow>
                                        <mi>z</mi>
                                    </mfrac>
                                </mrow>
                                <mo>)</mo>
                            </math></td><td>otherwise</td></tr><tr><td>JWD(s,t)</td><td style='background-color:#d4edda;'><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>JD(s,t)</mi><mo>+</mo><mi>p</mi><mo>x</mo><mi>q</mi><mo>x</mo><mo>(</mo><mn>1</mn><mo>-</mo><mi>JD(s,t)</mi><mo>)</mo></math></td><td style='background-color:#d4edda;'>All</td></tr></table>""",
                            unsafe_allow_html=True,
                        )
                    # st.markdown(
                    #     "<table style='text-align: center'><tr><th>Method</th><th>Equation</th><th>Condition</th></tr><tr><td rowspan='2'>JD(s,t)</td><td>0</td><td>if m = 0</td></tr><tr><td>1/3 (z/|s| + z/|t| + (z-y)/z)</td><td>otherwise</td></tr><tr><td>JWD(s,t)</td><td>JD(s,t) + p * q (1 - JD(s,t))</td><td>All</td></tr></table>",
                    #     unsafe_allow_html=True,
                    # )
                    st.write("")
                    st.write("List of abbreviations:")
                    st.markdown(
                        "<table style='text-align: center'><tr><th>Notation</th><th>Description</th></tr><tr><td>z</td><td>The number of matching characters</td></tr><tr><td>y</td><td>The number of transpositions divided by 2</td></tr><tr><td>s</td><td>Source string</td></tr><tr><td>t</td><td>Target string</td></tr><tr><td>|s|</td><td>Number of source string characters</td></tr><tr><td>|t|</td><td>Number of target string characters</td></tr><tr><td>p</td><td>The scaling factor, which by default is 0.1</td></tr><tr><td>q</td><td>The length of the matching prefixes</td></tr><tr><td>JD(s,t)</td><td>Jaro Similarity function</td></tr><tr><td>JWD(s,t)</td><td>Jaro winkler distance function</td></tr></table>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.write("Color description:")
                    st.markdown(
                        "<table style='text-align: center'><tr><th>Color</th><th>Description</th></tr><tr><td style='background-color:orange'></td><td>The length of the matching prefixes</td></tr><tr><td style='background-color:cyan'></td><td>The number of matching characters</td></tr><tr><td style='background-color:yellow'></td><td>The number of transpositions</td></tr></table>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.write("Result:")
                    st.write(f"z = {match_count}")
                    st.write(f"|s| = {len_str1}")
                    st.write(f"|t| = {len_str2}")
                    st.markdown(
                        f"""y = <math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>{transposition_count}</mn><mn>{2}</mn></mfrac>{transposition_count}/2 </math> = {transposition_count/2}""",
                        unsafe_allow_html=True,
                    )
                    st.write(f"p = {0.1}")
                    st.write(f"q = {prefix_length}")
                    if result != 0:
                        st.markdown(
                            f"""<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro Similarity between '{normalized_char1}' and '{normalized_char2}' = <math xmlns="http://www.w3.org/1998/Math/MathML">
                                <mfrac>
                                    <mn>1</mn>
                                    <mn>3</mn>
                                </mfrac>
                                <mo>(</mo>
                                <mrow>
                                    <mfrac>
                                        <mi>{match_count}</mi>
                                        <mi>{len_str1}</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mi>{match_count}</mi>
                                        <mi>{len_str2}</mi>
                                    </mfrac>
                                    <mo>+</mo>
                                    <mfrac>
                                        <mrow>
                                            <mi>{match_count}</mi>
                                            <mo>-</mo>
                                            <mi>{transposition_count/2}</mi>
                                        </mrow>
                                        <mi>{match_count}</mi>
                                    </mfrac>
                                </mrow>
                                <mo>)</mo>
                            </math> = <b>{round(result,3)}</b></div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro Similarity between '{normalized_char1}' and '{normalized_char2}' = <b>{0}</b></div>",
                            unsafe_allow_html=True,
                        )
                        # st.write(
                        #     f"JD('{normalized_char1},{normalized_char2}') = 1/3 ( ({match_count} / {len_str1}) + ({match_count} / {len_str2}) + ( ({match_count} - {0})/{match_count}) ) = {0}"
                        # )
                    # st.write(f"JWD(s,t) = JD(s,t) + p * q (1 - JD(s,t))")
                    st.markdown(
                        f"""<div style='background-color:#d4edda;padding: 20px;color: #155724;'>Jaro-Winkler Similarity between '{normalized_char1}' and '{normalized_char2}' = <math xmlns="http://www.w3.org/1998/Math/MathML"><mn>{round(jaro_score,3)}</mn><mo>+</mo><mn>{p}</mn><mo>x</mo><mn>{prefix_length}</mn><mo>x</mo><mo>(</mo><mn>{1}</mn><mo>-</mo><mn>{round(jaro_score,3)}</mn><mo>)</mo></math> = <b>{round(jaro_winkler_distance,3)}</b></div>""",
                        unsafe_allow_html=True,
                    )
                    if jaro_winkler_distance == 0:
                        st.warning("No matching characters!")
                # st.write(result)

if selected == "About":
    # st.title("Tentang Aplikasi")
    st.title("About the Application")

    # Informasi tentang aplikasi atau proyek
    # st.write(
    #     "Selamat datang di Aplikasi Streamlit About. Aplikasi ini dibuat sebagai contoh sederhana "
    #     "menggunakan Streamlit untuk membuat halaman 'about'."
    # )

    # Informasi tentang penulis atau tim
    # st.header("Sekilas Aplikasi")
    # st.write(
    #     "Spelling correction adalah proses identifikasi dan perbaikan kesalahan ejaan dalam sebuah teks. Aplikasi spelling correction membantu pengguna untuk menemukan kata-kata yang salah eja dan menawarkan saran perbaikan. Aplikasi ini memiliki 2 fitur. Fitur pertama digunakan untuk spelling correction Bahasa Madura. Sedangkan fitur kedua digunakan untuk perhitungan kemiripan atau kedekatan antara dua kata. Pada fitur kedua juga menyertakan detil perhitungan dari setiap metode."
    # )
    st.header("Application Overview")
    st.markdown(
        """
        <p>The task of spelling correction is to generate a top list of suggestions likely to be the expected correction of a misspelled word or string. A common way to measure the distance or similarity between a pair of strings is by comparing them directly, i.e., using a character-based string matching similarity method.</p>
        <P>Madurese is Indonesia’s regional language that has four accented characters (i.e., â, è, ḍ, and ṭ) in addition to the 26 Latin alphabets of the Indonesian language. They are commonly represented as Unicode of separate single-character using the Normalization Form Decomposition (NFD).</p>
        <P>This application is developed as part of the ongoing study to preserve the existence of Madurese. It offers two following features:</p>
        <ul>
            <li>Spelling correction: generate a top-5 list of correction suggestions for a misspelled (Madurese) string based on a character-based string matching similarity method</li>
            <li>String similarity: detail distance or similarity measurement between two (Madurese) strings based on a character-based string matching similarity method</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    # Kontak atau informasi lainnya
    # st.header("Metode Pada Aplikasi")
    st.header("Method on Application")
    st.markdown(
        """<div>
            This application offers five character-based string matching similarity methods:
            <ul>
                <li>Hamming Distance</li>
                <li>Levenshtein Distance</li>
                <li>Damerau-Levenshtein Distance</li>
                <li>Jaro Similarity</li>
                <li>Jaro-Winkler Similarity</li>
            </ul>
        </div>""",
        unsafe_allow_html=True,
    )
    st.header("Developer Team")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(
            "ibu_ifada.jpg",
            # caption="Noor Ifada",
            width=100,
        )
        st.markdown(
            "<div style='text-align: center; margin-left: 0px; width:100px'><a href='mailto:noor.ifada@trunojoyo.ac.id'>Noor Ifada</a></div>",
            unsafe_allow_html=True,
        )
        # st.markdown(
        #     "![noor idafa](https://github.com/ThoriqFathu/mecs/blob/main/ibu_Ifada.jpg)"
        # )
    with col2:
        st.image("bu_fika1.jpeg", width=100)
        st.markdown(
            "<div style='text-align: center; margin-left: 0px; width:100px'><a href='mailto:fika.rachman@trunojoyo.ac.id'>Fika Hastarita Rachman</a></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.image("ibu_sri1.jpeg", width=100)
        st.markdown(
            "<div style='text-align: center; margin-left: 0px; width:100px'><a href='mailto:s.wahyuni@trunojoyo.ac.id'>Sri Wahyuni</a></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.image("thoriq1.jpg", width=100)
        st.markdown(
            "<div style='text-align: center; margin-left: 0px; width:100px'><a href='mailto:thoriq771@gmail.com'>Muhammad Fathuthoriq</a></div>",
            unsafe_allow_html=True,
        )

    with col5:
        st.image("irul.jpeg", width=100)
        st.markdown(
            "<div style='text-align: center; margin-left: 0px; width:100px'><a href='mailto:moh.amirullah17@gmail.com'>Moh. Amirullah</a></div>",
            unsafe_allow_html=True,
        )
    # st.markdown("""<img src="thoriq.jpg">""", unsafe_allow_html=True)
    # st.header("Contact Us")
    # st.markdown(
    #     f"""<div>
    #     <ul>
    #     <li>Noor Ifada : <a href='noor.ifada@trunojoyo.ac.id'>noor.ifada@trunojoyo.ac.id</a></li>
    #     <li>Fika Hastarita Rachman : <a href='example@trunojoyo.ac.id'>example@trunojoyo.ac.id</a></li>
    #     <li>Sri Wahyuni : <a href='example@trunojoyo.ac.id'>example@trunojoyo.ac.id</a></li>
    #     <li>Muhammad Fathuthoriq : <a href='thoriq771@gmail.com'>thoriq771@gmail.com</a></li>
    #     <li>Moh. Amirullah : <a href='example@trunojoyo.ac.id'>example@trunojoyo.ac.id</a></li>
    #     </ul><div>""",
    #     unsafe_allow_html=True,
    # )
    st.header("Reference")
    st.markdown(
        f"""<div>
        <ul>
        <li>Ifada, N., Rachman, F. H., Wahyuni, S. (2023). Character-based String Matching Similarity Algorithms for Madurese Spelling Correction: A Preliminary Study. In <i> International Conference on Electrical Engineering and Informatics (ICEEI)</i> (pp. 1-6). IEEE. DOI: <a href='https://doi.org/10.1109/ICEEI59426.2023.10346716'>10.1109/ICEEI59426.2023.10346716</a></li>
        <li>Ifada, N., Rachman, F. H., Syauqy, M. W. M. A., Wahyuni, S., & Pawitra, A. (2023). MadureseSet: Madurese-Indonesian Dataset. <i> Data in Brief, 48,</i> 109035. DOI: <a href='https://doi.org/10.1016/j.dib.2023.109035'>10.1016/j.dib.2023.109035</a></li>
        </ul><div>""",
        unsafe_allow_html=True,
    )
