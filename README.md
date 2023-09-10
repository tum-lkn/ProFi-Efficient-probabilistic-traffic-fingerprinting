# Source code for ProFi: Scalable and efficient website fingerprinting

**Disclaimer**. The source code in this repository has been made publicly
available for transparency and as contribution for the scientific community. The
source code reflects in most parts the state in which the results for the referenced
publications were obtained. The source code has mostly been left as is.

This repository contains the source code for the publication:
> P. Krämer et al., “ProFi: Scalable and efficient website fingerprinting,” IEEE TNSM, pp. 1–14, Under Submission.


The code consists of three separate projects:

- tls-gatherer
- tls-fingerprint
- prototype

The `tls-gatherer` and the `tls-fingerprint` container contain docker files that
encapsulate the development environment.

## TLS Gatherer
The tls-gatherer project implements the measurement pipeline that we used to collect
our data. The project relies on a Kubernetes cluster and a backend-database, i.e., does not run standalone. A dump of the database is included in the provided data.


## TLS Fingerprint
The TLS Fingerprint repository contains the code to train and evaluate fingeprints. 
A good starting-point is the `scripts` folder. This folder contains Python scripts
that train and evaluate models, convert data, etc.
Note that the subfolders `baum-welch`, `argparse`, and `implementation/ctypes` 
contain C code that must be build manually.

To evaluate funcionality, build a docker image and mount the `tls-fingerprint` folder 
to `/opt/project` in the container. The data must be mounted separately (see below).


# Prototype
The prototype implements the entwork functions. The `start.py` scripts provides 
an interface to run the experiments.

Note that the prototype depends on OpenNetVM and MoonGen. This repository
contains only the implemented NFs.


