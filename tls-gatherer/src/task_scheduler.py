"""Scheldules tasks and orchestrates Docker instances"""
import socket
import loggingfactory
import docker
from typing import List, Any, Tuple
import time
import os

import docker_env.docker_instance
import config
import mysql.connector
import rediswq


class TagNotInDB(Exception):
    """Raised when a tag occurs that is not contained inside the database"""


class Supervisor():
    """Supervisies container instances"""
    def __init__(self):
        # pylint disable=unused-argument
        self.queue = {}  # dict of lists of lists
        self.containers = {}
        self.tasks = []
        self.database = None
        self.db_cursor = None
        self.pod_name = os.environ.get('POD_NAME')
        self.logger = loggingfactory.produce(f"Supervisor-{self.pod_name}")
        self.container_logger = loggingfactory.produce(f"Container-{self.pod_name}")
        try:
            self.client = docker.from_env()
            self.redis_q = rediswq.RedisWQ(name='gatherer', host='redis')
        except Exception as e:
            self.logger.exception(e)
            raise e
        self.logger.debug(f"Finished scheduler creation.")
        time.sleep(5)
    
    def connect_db(self):
        SWC_DB = mysql.connector.connect(host=config.SQL["host"],
                                 user=config.SQL["user"],
                                 passwd=config.SQL["passwd"],
                                 database=config.SQL["database"])
        self.database = SWC_DB
        self.db_cursor = self.database.cursor(buffered=True)

    def disconnect_db(self):
        self.database.close()
        self.db_cursor = None

    def retrieve_url_id(self, job_id):
        """in: job_id
           out: url id"""
        query = "select url, browser from jobs where id = (%s)" % job_id
        self.logger.debug(f"Execute query: {query}")

        self.db_cursor.execute(query)
        url_id, browser = self.db_cursor.fetchall()[0]
        return url_id, browser

    def retrieve_job_details(self, job_id):
        """returns job details for a given url_id"""
        url_id, browser = self.retrieve_url_id(job_id)
        query = "SELECT id, url, company, tags from urls where id = (%s)" % url_id
        self.logger.debug(f"Execute query: {query}")

        self.db_cursor.execute(query)
        row = self.db_cursor.fetchall()[0]
        if browser not in config.DOCKER_AVAILABLE_BROWSERS:
            return None
        else:
            job = {"url_id": row[0],
                   "job_id": job_id,
                   "url": str(row[1]).replace(' ', '%20'),
                   "company_id": row[2],
                   "tags": str(row[3]).split(";"),
                   "browser": browser,
                   }
            return job

    def initiate(self):
        """Intended for async operation, for now only starts the first job"""

        self.logger.info('TLS Gatherer initiating.')
        self.logger.info("initiating")

        reported_queue_empty = False
        while True:
            try:
                if self.redis_q.empty():
                    if not reported_queue_empty:
                        self.logger.info("Queue is empty for worker with sessionID: {}".format(self.redis_q.sessionID()))
                        reported_queue_empty = True
                    time.sleep(10)
                else:
                    self.logger.info("Retrieved Item")
                    item = self.redis_q.lease(lease_secs=30, block=True, timeout=2)
                    reported_queue_empty = False
                    if item is not None:
                        db_id = int(item.decode("utf-8"))
                        self.logger.info(f"DB ID retrieved: {db_id}")
                        self.redis_q.complete(item)
                        self.logger.debug("Connect to db.")
                        self.connect_db()
                        job = self.retrieve_job_details(db_id)
                        self.logger.info(f"job info retrieved: {str(job)}")
                        self.create_container(job)
                        self.logger.info("job finished")
                        self.disconnect_db()
                    else:
                        time.sleep(30)
            except Exception as e:
                self.logger.error(f"Failed to execute Job; {e}")

    def create_container(self, job):
        """processes queue, creates instance of container class for each trace"""
        # url_id = next(iter(self.queue))
        # url = self.queue[url_id]["url"]
        # tags = self.queue[url_id]["tags"]
        # current_browser = next(iter(self.queue[url_id]["browsers_to_trace"]))
        url_id = job['url_id']
        url = job["url"]
        tags = job["tags"]
        current_browser = job["browser"]
        idx = int(job['job_id'])
        self.logger.info("\tcreating container")
        self.containers[idx] = docker_env.docker_instance.Container(
            tag=config.DOCKER_IMAGE_TAG,
            client=self.client,
            host_traces_dir=config.HOST_TRACES_DIR,
            container_traces_dir=config.CONTAINER_TRACES_DIR,
            url=url,
            tags=tags,
            browser=current_browser,
            url_id=url_id,
            logger=self.container_logger
        )

        self.logger.info("\tContainer Created, set ipv4 and build image")
        try:
            self.containers[idx].ipv4 = dns_lookup(self.containers[idx].url)
        except Exception as e:
            self.logger.error(f"DNS Lookup failed for URL {self.containers[idx].url}")
            self.logger.exception(e)
            self.containers[idx].ipv4 = 'unknown'

        self.containers[idx].build_image()
        self.logger.info("\tImage builded, start container.")
        self.containers[idx].run()
        self.logger.info("\tcontainer ran, start capture")
        # await self.containers[idx].capture(1, "wget") async stuff
        self.containers[idx].capture()
        self.remove_container(idx)
        self.logger.info("\tCaptured, upload metadata")
        try:
            self.upload_metadata(idx)
            self.logger.info("\tMetadata uploaded")
        except Exception as e:
            self.logger.error("Failure during metadata upload.")
            self.logger.exception(e)

    def make_tag_assignment(self, idx):
        self.logger.info("Assign tags to trace.")
        query = f"SELECT id from traces_metadata WHERE filename = '{self.containers[idx].filename}'"
        self.db_cursor.execute(query)
        trace_id = self.db_cursor.fetchall()[0][0]

        # adding new tags to DB
        tag_ids = []
        self.logger.debug("Try to retrieve following tags: {}".format(str(self.containers[idx].tags)))
        for tag in self.containers[idx].tags:
            query = f"SELECT id FROM tags WHERE tag = '{tag}'"
            self.logger.debug(f"Execute query: {query}")
            self.db_cursor.execute(query)
            records = self.db_cursor.fetchall()
            if self.db_cursor.rowcount:
                tag_ids.append(records[0][0])
            else:
                self.logger.info(tag)
                raise TagNotInDB()

                # only required if new tags should be added
                # query = "INSERT INTO tags (id, tag) VALUES (NULL, %s)"
                # db_cursor.execute(query, (tag,))
                # self.database.commit()
                # decided against LAST_INSERT_ID() because of multiprocessing
                # query = f"SELECT id FROM tags WHERE tag = '{tag}'"
                # db_cursor.execute(query)
                # records = db_cursor.fetchall()
                # tag_ids.append(records[0][0])"""

        # linking trace with tags
        for tag_id in tag_ids:
            query = "INSERT INTO tag_assignment (id, tag_fk, trace_fk) VALUES (NULL, %s, %s)"
            self.logger.debug(f"Execute query: {query}")
            val = (tag_id, trace_id)
            self.db_cursor.execute(query, val)
            self.database.commit()

    def upload_metadata(self, idx):
        """Saves metadata in SQL DB, adds new tags to tags_table
        and links traces to tags"""

        query = """INSERT INTO traces_metadata
                   (id, url, ipv4, filename, start_capture, end_capture,
                   browser, os, tls_version, is_resumption, was_successful, added_by, ssl_key_filename)
                   VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        self.logger.debug(f"Execute query {query}")
        val = (self.containers[idx].url_id, self.containers[idx].ipv4, self.containers[idx].filename,
               self.containers[idx].start, self.containers[idx].end, self.containers[idx].browser_string,
               self.containers[idx].operatingsystem,
               'NULL' if self.containers[idx].tls_version is None else self.containers[idx].tls_version, 
               0, self.containers[idx].was_successful, config.USER, self.containers[idx].ssl_key_filename)
        self.logger.debug("with values {}".format(', '.join([str(x) for x in val])))

        self.db_cursor.execute(query, val)
        self.database.commit()
        self.logger.info("added trace to db")
        # self.make_tag_assignment(idx)

        execution_state = self.containers[idx].execution_state
        query = f"UPDATE `jobs` SET `state` = '{execution_state}', `last_status_update` = '{self.containers[idx].end}' WHERE `jobs`.`id` = {idx};"
        self.logger.debug(f"Execute query: {query}")
        self.db_cursor.execute(query)
        self.database.commit()
        self.logger.debug("Updated job info.")

    def remove_container(self, idx):
        """Removes docker container and triggers replemishment if queue not empty"""
        self.logger.debug("remove container")
        self.containers[idx].container.remove(force=True)
        self.logger.info("container removed")
        time.sleep(1)


def dns_lookup(url):
    """extracts tld-part of url and resolves it"""
    if "//" in url:
        url = url.split("//")[1]
    if "/" in url:
        url = url.split("/")[0]
    if ':' in url:
        url = url.split(':')[0]
    return socket.gethostbyname(url)


if __name__ == "__main__":
    C_MANAGER = Supervisor()
    C_MANAGER.initiate()
