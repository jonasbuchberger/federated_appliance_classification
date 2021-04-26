#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: René Schwermer
E-Mail: rene.schwermer@tum.de
Date: 20.10.2020
Project: Federated Learning Toolbox
Version: 0.0.1
"""
import os
import time

from keystoneauth1 import session
from keystoneauth1.identity import v3
from novaclient import client
from novaclient.exceptions import NotFound


class OpenStackConnector:
    """
    Functions to interact with an openstack interface.
    """

    def __init__(self,
                 username: str,
                 pwd: str):
        self.username = username
        self.pwd = pwd
        self.nova = self.connection_helper()

    def connection_helper(self):
        """

        """
        auth = v3.Password(
            auth_url='https://openstack.cluster.msrg.in.tum.de:5000/v3/',
            username=self.username,
            password=self.pwd,
            project_domain_name='default',
            user_domain_name='default')

        sess = session.Session(auth=auth)
        website_cluster = "https://openstack.cluster.msrg.in.tum.de:"
        nova_client = client.Client(2,
                                    session=sess,
                                    endpoint_override=website_cluster +
                                                      "8774/v2.1/")
        n2 = client.Client(2,
                           session=sess,
                           endpoint_override=website_cluster + "9292")
        nova_client.glance = n2.glance
        return nova_client

    def create_instance(self,
                        server_name='started-with-api',
                        image='kvm-ubuntu-xenial',
                        flavor='m1.small',
                        availability_zone='kvm-ssd',
                        key=None) -> tuple:
        """

        """
        nflavor = self.nova.flavors.find(name=flavor)
        nimage = self.nova.glance.find_image(image)

        if key is None:
            nkeys = self.nova.keypairs.list()
            if len(nkeys) > 0:
                nkey = self.nova.keypairs.list()[0]
            else:
                raise Exception('no-key', 'no key is available in nova')
        else:
            nkey = self.nova.keypairs.find('key')
            if nkey is None:
                raise Exception('no-key',
                                'no key with that name is '
                                'found in nova')

        new_server = self.nova.servers.create(server_name,
                                              flavor=nflavor,
                                              image=nimage,
                                              key_name=nkey.name,
                                              availability_zone=availability_zone)

        while new_server.status != "ACTIVE":
            time.sleep(0.1)
            new_server = self.nova.servers.get(new_server.id)

        interface = new_server.interface_list()[0]
        while interface.port_state != 'ACTIVE':
            time.sleep(0.1)
            new_server = self.nova.servers.get(new_server.id)
            interface = new_server.interface_list()[0]

        fixed_ip = interface.fixed_ips[0]
        ip_address = fixed_ip['ip_address']

        return ip_address

    def instance_created(self,
                         server_name: str) -> bool:
        """
        Checks if a given server / instance exists on the connected
        open stack acoount.

        :param server_name: str, Name of server / instance in open stack
        """
        instances = self.get_instances()

        for server in instances:
            if server_name == server.name:
                return True
            else:
                return False

    def how_many_instances(self,
                           server_name: str) -> int:
        """
        Counts how often a given server / instance exists on the
        connected open stack account.

        :param server_name: str, Name of server / instance in open stack
        """
        instances = self.get_instances()
        count = 0
        for server in instances:
            if server_name == server.name:
                count += 1

        return count

    def is_instance_reachable(self,
                              ip_address: str) -> bool:
        """
        Pings a given ip address once and evaluates the output. This
        works only on Linux and maybe MacOS.

        :param ip_address: str, ip address to ping
        """
        response = os.system("ping -c 1 " + ip_address)
        if response == 0:
            return True
        else:
            return False

    def get_instances(self) -> list:
        """
        Lists all available instances on the connected
        openstack account.
        """
        return self.nova.servers.list()

    def get_instance_names(self) -> list:
        """

        :return: list,
        """
        instances = self.get_instances()

        names = []
        for server in instances:
            names.append(server.name)

        return names

    def get_flavors(self) -> list:
        """
        Lists all available flavors on the connected
        openstack account.
        """
        return self.nova.flavors.list()

    def find_flavor_by_ram(self,
                           ram: int):
        """
        Searches through the available flavors at the connected
        openstack account. If it is not a complete match between the
        given ram and the one in the flavor an error is thrown.

        :param ram: int, available ram in a flavor
        """
        try:
            flavor = self.nova.flavors.find(ram=ram)
            return flavor

        except NotFound:
            print("No flavor found with %s ram" % ram)
            return None

    def check_for_duplicates(self,
                             vm_names: list):
        """

        :param vm_names: list,
        """
        names = self.get_instance_names()
        duplicates = list(set([x for x in names if names.count(x) > 1]))

        [print("Virtual machine " + name +
               " exists more than once; stored most recent ip address.")
         if name in duplicates else print("") for name in vm_names]
