#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:37:03 2019

@author: chr
"""
import psutil
import logging
import threading
import netifaces as nif
from sys import platform
from datetime import datetime


class skype_monitoring:
    """
    This class scans for connection changes for the process_name.
    When a skype call is active a UDP connection will (as standard) be
    established (This is unless its restriced).

    As such if a new UDP connection is established for a skype process then
    it can be expted that a skype call is in progress and a time log is
    created.

    Note this might not work with things skype for business or other special
    cases.
    """

    def __init__(self, interval=.5, first_interval=0):
        """
        Parameter setup for skypemonitoring.
        Using the system platform information the appropriate skype process
        is identified as well as any network interfaces MAC addresses.
        This will further start the logging using the logging module.
        The first element of the log is the network interface and MAC addresses
        of the device. Futhermore, the logger saves the time of creation.
        (LOCAL)

        >>> platform = 'linux'
        >>> test = skype_monitoring()
        >>> test.processname
        "skypeforlinux"
        """
        self.timer = None
        self.interval = interval
        self.first_interval = first_interval
        self.running = False
        self.is_started = False
        self.skype_active = False

        if platform == 'linux' or platform == 'linux2':
            self.process_name = "skypeforlinux"
        elif platform == 'win32':
            self.process_name = "skype.exe"
        else:
            raise NotImplementedError("Mac isn't implemented")
        FileName = 'SkypeCall' + datetime.utcnow().strftime('%d%m%H') + '.log'
        addrs = {i: nif.ifaddresses(i)[nif.AF_LINK][0]['addr']
                 for i in nif.interfaces()}
        logging.basicConfig(filename=FileName,
                            level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
        logging.info('Network interface and MAC addrs:' + str(addrs))

        thread = threading.Thread(target=self.first_start)
        thread.start()

    def mon(self, kind='udp'):
        """
        The monitor function.
        This function scans for UDP connection changes for self.process_name.
        This function utilise psutils connection function as such please reffer
        to the psutil documentation.

        # TODO Investigate the 5'th created child process (recursive).

        Parameters
        ----------
        Kind : str
            The connection type to monitor. (default: 'udp')
        """
        udp_count = 0
        for process in psutil.process_iter():
            # get all process info in one shot
            with process.oneshot():
                if process.name() == self.process_name:
                    udp_count += len(process.connections(kind=kind))
        if not self.skype_active and udp_count > 1:
            self.skype_active = True
            logging.info(f'Skype call started (UDP connection established)\
 {udp_count}')
        elif self.skype_active and udp_count > 1:
            pass
        elif self.skype_active and udp_count <= 1:
            self.skype_active = False
            logging.info(f'Skype call terminated (UDP connection terminated)\
 {udp_count}')

    def first_start(self):
        try:
            # no race-condition here because only control thread will call this
            # if already started will not start again
            if not self.is_started:
                self.is_started = True
                self.timer = threading.Timer(self.first_interval, self.run)
                self.running = True
                self.timer.start()
        except Exception as e:
            logging.error(f'timer first start failed {e.massage}')
            raise

    def run(self):
        # if not stopped start again
        if self.running:
            self.timer = threading.Timer(self.interval, self.run)
            self.timer.start()
        self.mon()

    def stop(self):
        # cancel current timer in case failed it's still OK
        # if already stopped doesn't matter to stop again
        if self.timer:
            self.timer.cancel()
        self.running = False


if __name__ == "__main__":
    test = skype_monitoring()
