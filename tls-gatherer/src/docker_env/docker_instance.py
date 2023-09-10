"""Manages a Docker container instance"""
import time
import os
import hashlib
import loggingfactory


class BrowserQueryUnsuccessful(Exception):
    """Raised when Exception occurs while URL query"""


class TraceEmpty(Exception):
    """Raised when trace size is unreasonalby small after trace"""


class Container:
    """Manages a container and contains its config"""

    def __init__(self, tag, client, host_traces_dir, container_traces_dir, url, tags, browser,  # pylint: disable=too-many-arguments
                 url_id, logger=None):  # pylint: disable=too-many-arguments
        self.docker_tag = tag
        self.tags = tags
        self.client = client
        self.filename = None
        self.container = None
        self.traces_dir = host_traces_dir
        self.container_traces_dir = container_traces_dir
        self.url = url
        self.start = None
        self.end = None
        self.browser = browser
        self.browser_string = None
        self.operatingsystem = None
        self.ipv4 = None
        self.tls_version = None
        self.ssl_key_filename = None
        self.url_id = url_id
        self.was_successful = None
        self.trace_for_seconds = 7
        self.pod_name = os.environ.get('POD_NAME')
        self.filename = self.generate_unique_filename()
        self.logger = loggingfactory.produce(f"Container-{self.pod_name}") if logger is None else logger
        self.execution_state = 'finished'

    def generate_unique_filename(self):
        """does what it says based on time and hashing"""
        zeitstempel = time.time()
        haesh = str(int(hashlib.sha256(str(zeitstempel).encode('utf-8')).hexdigest(), 16) % 10**8)

        # extract tld-part:
        tld = self.url.split("//")[1].split("/")[0].split(".")[-2]
        return f"{tld}_{self.browser}_{self.pod_name}_{str(haesh)[0:12]}"

    def build_image(self):
        """Builds base image with all browsers installed."""
        self.client.images.build(path="/mnt/host/docker_env/", tag=self.docker_tag)

    def run(self):
        """Starts container and mounts host volume"""
        self.ssl_key_filename = f'ssl-keys-{self.pod_name}.log'
        self.container = self.client.containers.run(
            image=self.docker_tag,
            auto_remove=True,
            privileged=True,
            detach=True,
            tty=True,
            stdin_open=True,
            cap_add=["NET_RAW", "NET_ADMIN"],
            volumes={f"{self.traces_dir}": {"bind": "/traces/"}, '/tmp/.X11-unix': {"bind": "/tmp/.X11-unix"}},
            environment=["DISPLAY=$DISPLAY", f"SSLKEYLOGFILE=/traces/{self.ssl_key_filename}"]
        )
        _, self.operatingsystem = self.container.exec_run("uname -r")

    def firefox_query(self):
        """uses firefox cli to load requested url in headless mode
        """
        _, self.browser_string = self.container.exec_run("firefox --version")
        self.container.exec_run(f"tmux new-session -d -s 'firefox' firefox -headless -new-tab {self.url}")
        time.sleep(self.trace_for_seconds)
        self.container.exec_run("tmux kill-session -t firefox")

    def chromium_query(self):
        """uses headless chromium-browser to load requested url populates browser_string attribute for chromium """
        _, browser_version = self.container.exec_run("chromium-browser --version")
        browser_parts = str(browser_version, "utf-8").split(" ")
        self.browser_string = f"{browser_parts[0]} {browser_parts[1]}"

        self.container.exec_run(f"chromium-browser {self.url} -headless --no-sandbox")
        time.sleep(self.trace_for_seconds)

    def opera_query(self):
        """uses headless chromium-browser to load requested url populates browser_string attribute for chromium """
        _, browser_version = self.container.exec_run("opera --version")
        browser_version = str(browser_version)
        self.browser_string = f"Opera {browser_version}"

        self.container.exec_run(f"opera {self.url} -headless --no-sandbox")
        time.sleep(self.trace_for_seconds)

    def microsoft_edge_query(self):
        """uses headless chromium-browser to load requested url populates browser_string attribute for chromium """
        _, browser_version = self.container.exec_run("microsoft-edge-dev --version")
        browser_parts = str(browser_version, "utf-8").split(" ")
        self.browser_string = f"{browser_parts[0]} {browser_parts[1]} {browser_parts[2]}"

        self.container.exec_run(f"microsoft-edge-dev {self.url} --no-sandbox")
        time.sleep(self.trace_for_seconds)

    def wget_query(self):
        """downloads page index using wget and populates browser_string attribute"""
        _, bytestring = self.container.exec_run("wget --version")
        browser_parts = str(bytestring, "utf-8").split(" ")

        self.browser_string = f"{browser_parts[0]} {browser_parts[1]} {browser_parts[2]}"
        output = self.container.exec_run(f"wget {self.url}")
        self.was_successful = not output[0]
        if not output[0] == 0:
            self.logger.error(f"WGET-Browser returned non-zero state for URL: {self.url}\n{output[1]}")
            # raise BrowserQueryUnsuccessful
        time.sleep(self.trace_for_seconds)

    def capture(self, device='eth0'):
        """initiates pcapng capture using tshark"""
        self.logger.info("Disable offloading features.")
        self.container.exec_run(f"ethtool -K {device} tx off sg off tso off gro off gso off")
        _, output = self.container.exec_run(f"ethtool -k {device}")
        self.logger.debug("NIC features before tracing are: ")
        self.logger.debug(output.decode('utf8'))

        self.logger.info("Capture of %s starting", self.url)
        self.container.exec_run(f"tmux new-session -d -s 'cap' tcpdump -w /traces/{self.filename}.pcapng")
        # timestamp trace start
        now = time.localtime()
        self.start = time.strftime('%Y-%m-%d %H:%M:%S', now)
        time.sleep(1)
        self.logger.info(f"Generating traffic using {self.browser}")

        try:
            if self.browser == "firefox":
                self.firefox_query()
            if self.browser == "wget":
                self.wget_query()
            if self.browser == "chromium":
                self.chromium_query()
            if self.browser == 'opera':
                self.opera_query()
            if self.browser == 'edge':
                self.microsoft_edge_query()
        except Exception as e:
            self.logger.error(f"Failure during traffic generation with {self.browser}")
            self.logger.exception(e)
            self.execution_state = 'traffic generation failed'
            return

        self.logger.debug("Kill tmux session")
        self.container.exec_run("tmux kill-session -t cap")

        # timestamp trace end
        now = time.localtime()
        self.end = time.strftime('%Y-%m-%d %H:%M:%S', now)
        self.logger.info(f"Trace stopped for {self.url}")

        if os.path.exists(os.path.join(self.container_traces_dir, f"{self.filename}.pcapng")):
            self.logger.debug("Capture Exists", f"{self.container_traces_dir}{self.filename}.pcapng")
        else:
            self.logger.debug("Capture does not exist", f"{self.container_traces_dir}{self.filename}.pcapng")

        try:
            if os.stat(os.path.join(self.container_traces_dir, f"{self.filename}.pcapng")).st_size < 2000:
                self.execution_state = 'Trace empty'
                self.logger.error(f"Trace empty for url {self.url} and browser {self.browser}")
                return
                # raise TraceEmpty
        except Exception as e:
            self.execution_state = 'trace size check failed'
            self.logger.error("Error during check of trace size.")
            self.logger.exception(e)
            return
        self.logger.debug(f"After trace size check {self.url}")

        # fetch tls version
        try:
            self.logger.debug("Fetch TLS Version")
            self.fetch_tls_version()
            self.logger.debug("TLS Version fetched")
        except Exception as e:
            self.logger.error("Unexpected error during TLS version check.")
            self.logger.exception(e)
            self.tls_version = None
            self.execution_state = 'tls version check failed'

    def fetch_tls_version(self):
        """finds out the tls version used by application using thark"""
        tls_grab = f"""tshark -r /traces/{self.filename}.pcapng -T fields
        -e _ws.col.Protocol -E header=y -E separator=, -E quote=d"""
        _, out = self.container.exec_run(tls_grab)
        outlines = out.decode("utf-8").split("\n")
        # filter out duplicate lines from list
        no_duplicates = list(dict.fromkeys(outlines[2:]))
        # filter out lines that arent valid TLS versions
        tls_versions = [s for s in no_duplicates if "TLSv1." in s]
        # remove quotes from string
        tls_versions = [s.replace('"', '') for s in tls_versions]
        # keeping only the version of the first tls connection made
        self.tls_version = tls_versions[0]
