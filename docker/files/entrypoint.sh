#!/bin/bash
service mysql start
service slurmdbd start
service slurmctld start
service slurmd start

/usr/sbin/sshd -D
