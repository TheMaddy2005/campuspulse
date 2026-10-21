# Data

This project uses the **Student Performance** dataset from the UCI Machine
Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/320/student+performance
- License: CC BY 4.0
- Citation: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary
  School Student Performance*. In A. Brito and J. Teixeira (Eds.), Proceedings
  of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, 2008.

## Files

- `raw/student-mat.csv` - 395 students, Mathematics course grades (used for training)
- `raw/student-por.csv` - 649 students, Portuguese language grades (not used, kept for reference)
- `raw/student.txt` - original attribute description from the dataset authors

Each row is one student. Features cover demographics (age, address, family),
study habits (study time, past failures, absences), lifestyle (going out,
alcohol consumption) and school support. The target is `G3`, the final grade
(0-20). `G1` and `G2` are the first and second period grades.

To re-download the raw data instead of using the committed copy:

    python scripts/download_data.py
