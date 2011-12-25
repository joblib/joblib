# -*- coding: utf-8 -*-
"""
    sphinx_pypi_upload
    ~~~~~~~~~~~~~~~~~~

    setuptools command for uploading Sphinx documentation to PyPI

    :author: Jannis Leidel
    :contact: jannis@leidel.info
    :copyright: Copyright 2009, Jannis Leidel.
    :license: BSD, see LICENSE for details.

Modified for joblib by Gael Varoquaux
"""


import os
import socket
try:
    import httplib
    import urlparse
    from cStringIO import StringIO as BytesIO
except ImportError:
    # Python3k
    import http as httplib
    from urllib import parse as urlparse
    from io import BytesIO
import base64

from distutils import log
from distutils.command.upload import upload


class UploadDoc(upload):
    """Distutils command to upload Sphinx documentation."""

    description = 'Upload Sphinx documentation to PyPI'
    user_options = [
        ('repository=', 'r',
         "url of repository [default: %s]" % upload.DEFAULT_REPOSITORY),
        ('show-response', None,
         'display full response text from server'),
        ('upload-file=', None, 'file to upload'),
        ]
    boolean_options = upload.boolean_options

    def initialize_options(self):
        upload.initialize_options(self)
        self.upload_file = None

    def finalize_options(self):
        upload.finalize_options(self)
        if self.upload_file is None:
            self.upload_file = 'doc/documentation.zip'
        self.announce('Using upload file %s' % self.upload_file)

    def upload(self, filename):
        content = open(filename, 'rb').read()
        meta = self.distribution.metadata
        data = {
            ':action': 'doc_upload',
            'name': meta.get_name(),
            'content': (os.path.basename(filename), content),
        }
        # set up the authentication
        auth = "Basic " + base64.encodestring(self.username + ":" + \
                self.password).strip()

        # Build up the MIME payload for the POST data
        boundary = '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254'
        sep_boundary = '\n--' + boundary
        end_boundary = sep_boundary + '--'
        body = BytesIO()
        for key, value in data.items():
            # handle multiple entries for the same name
            if type(value) != type([]):
                value = [value]
            for value in value:
                if type(value) is tuple:
                    fn = ';filename="%s"' % value[0]
                    value = value[1]
                else:
                    fn = ""
                value = str(value)
                body.write(sep_boundary)
                body.write('\nContent-Disposition: form-data; name="%s"' % key)
                body.write(fn)
                body.write("\n\n")
                body.write(value)
                if value and value[-1] == '\r':
                    body.write('\n')  # write an extra newline (lurve Macs)
        body.write(end_boundary)
        body.write("\n")
        body = body.getvalue()

        self.announce("Submitting documentation to %s" % (self.repository),
                      log.INFO)

        # build the Request
        # We can't use urllib2 since we need to send the Basic
        # auth right with the first request
        schema, netloc, url, params, query, fragments = \
            urlparse.urlparse(self.repository)
        assert not params and not query and not fragments
        if schema == 'http':
            http = httplib.HTTPConnection(netloc)
        elif schema == 'https':
            http = httplib.HTTPSConnection(netloc)
        else:
            raise AssertionError("unsupported schema " + schema)

        data = ''
        loglevel = log.INFO
        try:
            http.connect()
            http.putrequest("POST", url)
            http.putheader('Content-type',
                           'multipart/form-data; boundary=%s' % boundary)
            http.putheader('Content-length', str(len(body)))
            http.putheader('Authorization', auth)
            http.endheaders()
            http.send(body)
        except socket.error as e:
            self.announce(str(e), log.ERROR)
            return

        response = http.getresponse()
        if response.status == 200:
            self.announce('Server response (%s): %s' %
                          (response.status, response.reason), log.INFO)
        elif response.status == 301:
            location = response.getheader('Location')
            if location is None:
                location = 'http://packages.python.org/%s/' % meta.get_name()
            self.announce('Upload successful. Visit %s' % location,
                          log.INFO)
        else:
            self.announce('Upload failed (%s): %s' % \
            (response.status, response.reason), log.ERROR)
        if self.show_response:
            print('-' * 75 + response.read() + '-' * 75)

    def run(self):
        zip_file = self.upload_file
        self.upload(zip_file)
