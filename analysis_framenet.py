import os
import re

import duckdb


def init():
    duckdb.sql("CREATE TABLE metrics("
               "    model VARCHAR,"
               "    layer INT,"
               "    type3 VARCHAR,"
               "    \"group\" VARCHAR,"
               "    prefix VARCHAR,"
               "    epoch INT,"
               "    type1 VARCHAR,"
               "    type2 VARCHAR,"
               "    name VARCHAR,"
               "    metric VARCHAR,"
               "    value FLOAT,"
               "    status VARCHAR GENERATED ALWAYS AS (regexp_extract(name, '(asserted|implied)')),"
               "    name2 VARCHAR GENERATED ALWAYS AS (regexp_replace(name, '(True|False)', '')),"
               "    \"f/fe\" VARCHAR GENERATED ALWAYS AS (regexp_extract(name, '(f_|fe_)')),"
               ");")

    for filename in os.listdir('.'):
        match = re.fullmatch(r'metrics_(.+)_(\d+)_?(.*)?\.csv', filename)

        if match is not None:
            model = match.group(1)
            layer = match.group(2)
            type3 = match.group(3)
            query = \
                "INSERT INTO metrics "\
                "SELECT '{}' AS model, '{}' AS layer, '{}' AS type3, * "\
                "FROM read_csv_auto('{}', header=true)".format(model, layer, type3, filename)
            duckdb.sql(query)


def sv_sat_summary():
    duckdb.sql(
        "COPY ("
        "   PIVOT ("
        "       SELECT * FROM metrics"
        "       WHERE metric='sat' AND prefix IS NULL AND \"group\"='test' AND type2='sv'"
        "   ) ON layer, type3 USING AVG(value) AS avg, COUNT(value) as cnt"
        "   GROUP BY model, 'f/fe', status"
        ") TO 'analysis_sv_sat_summary.csv' (HEADER, DELIMITER ',')"
    )


def cf_sat_summary():
    duckdb.sql(
        "COPY ("
        "   PIVOT ("
        "       SELECT * FROM metrics"
        "       WHERE metric='sat' AND prefix IS NULL AND \"group\"='test' AND epoch=-2"
        "   ) ON layer, type3 USING AVG(value) AS avg, COUNT(value) as cnt"
        "   GROUP BY model, type2"
        ") TO 'analysis_cf_sat_summary.csv' (HEADER, DELIMITER ',')"
    )


def frame_sat():
    duckdb.sql(
        "COPY ("
        "   PIVOT ("
        "       SELECT * FROM metrics"
        "       WHERE metric='sat' AND prefix IS NULL AND \"group\"='test' AND type1='cf' AND type2='sv'"
        "   ) ON layer, type3 USING AVG(value) AS avg, COUNT(value) as cnt"
        "   GROUP BY model, 'f/fe', name2"
        ") TO 'analysis_frame_sat.csv' (HEADER, DELIMITER ',')"
    )


def frame_f1():
    duckdb.sql(
        "COPY ("
        "   PIVOT ("
        "       SELECT * FROM metrics"
        "       WHERE metric='f1' AND prefix IS NULL AND \"group\"='test'"
        "   ) ON layer, type3 USING AVG(value) AS avg, COUNT(value) as cnt"
        "   GROUP BY model, status, name"
        ") TO 'analysis_frame_f1.csv' (HEADER, DELIMITER ',')"
    )


if __name__ == "__main__":
    init()

    sv_sat_summary()
    cf_sat_summary()

    frame_sat()
    frame_f1()

