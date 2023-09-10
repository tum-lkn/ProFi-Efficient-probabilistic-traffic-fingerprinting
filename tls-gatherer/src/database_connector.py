"""connection management to SQL DB"""

import mysql.connector
import config

SWC_DB = mysql.connector.connect(host=config.SQL["host"],
                                 user=config.SQL["user"],
                                 passwd=config.SQL["passwd"],
                                 database=config.SQL["database"])
