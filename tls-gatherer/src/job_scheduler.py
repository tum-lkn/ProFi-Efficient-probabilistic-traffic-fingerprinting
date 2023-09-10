import config
import database_connector
import time
import redis
import loggingfactory


logger = loggingfactory.produce("JobScheduler")
logger.info("Job Scheduler initiating")
redis_db = redis.StrictRedis(host=config.REDIS_HOST)


db_cursor = database_connector.SWC_DB.cursor(buffered=True)
query = "SELECT id, url, company, tags from urls"
db_cursor.execute(query)
rows = db_cursor.fetchall()

now = time.strftime('%Y-%m-%d %H:%M:%S')

for browser in config.JOB_SCHEDULED_BROWSERS:
    for row in rows:
        try:
            query = f"INSERT INTO `jobs` (`id`, `url`, `browser`, `state`, `error report`, `scheduled_date`, `last_status_update`) VALUES (NULL, {row[0]}, '{browser}', 'scheduled', '', '{now}', '{now}'); "
            print(db_cursor.execute(query))
            database_connector.SWC_DB.commit()

            query = "SELECT LAST_INSERT_ID(); "
            db_cursor.execute(query)
            fetched = db_cursor.fetchall()
            redis_db.lpush(config.REDIS_QUEUE, fetched[0][0])
        except Exception as e:
            logger.error(e)
logger.info(f"Scheduled {2*len(rows)} jobs successfully")
