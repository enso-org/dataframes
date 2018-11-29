FROM centos:7

RUN yum install -y centos-release-scl yum-utils wget git xz gperf bison flex libffi-devel curl-devel
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum install -y devtoolset-7

# add Dev-Toolset to the environment variables (so everyone can be happy with GCC-7)
ENV PATH="/opt/rh/devtoolset-7/root/usr/bin:${PATH}" \
    MANPATH="/opt/rh/devtoolset-7/root/usr/share/man" \
    INFOPATH="/opt/rh/devtoolset-7/root/usr/share/info" \
    PCP_DIR="/opt/rh/devtoolset-7/root" \
    PERL5LIB="/opt/rh/devtoolset-7/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-7/root/usr/lib/perl5:/opt/rh/devtoolset-7/root//usr/share/perl5/vendor_perl" \
    LD_LIBRARY_PATH="/opt/rh/devtoolset-7/root:/usr/lib64:/usr/local/lib:/usr/local/lib64" \
    PYTHONPATH="/opt/rh/devtoolset-7/root/usr/lib64/python2.6/site-packages:/opt/rh/devtoolset-7/root/usr/lib/python2.6/site-packages"

ENV PATH="/root/.local/bin:${PATH}"

# needed for CMake scripts relying on pkg-config
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

WORKDIR /

# openssl, needed to be installed before building Python
RUN wget https://www.openssl.org/source/openssl-1.1.1.tar.gz &&\
    tar -xzf openssl-1.1.1.tar.gz &&\
    cd openssl-1.1.1 &&\
    ./config shared &&\
    make -j 16 &&\
    make install &&\
    cd .. &&\
    rm -rf openssl-1.1.1.tar.gz openssl-1.1.1

# Zlib, needs to be installed before building Python
RUN wget https://zlib.net/zlib-1.2.11.tar.gz &&\
    tar -xzf zlib-1.2.11.tar.gz &&\
    cd zlib-1.2.11 &&\
    ./configure && make -j 8 install &&\
    cd .. &&\
    rm -rf zlib-1.2.11.tar.gz zlib-1.2.11

# install Python - it requires zlib and libffi
# using pip requires libssl and it must be locatable in LD_LIBRARY_PATH (or perhaps through the configure flag)
# having /python-dist/lib in LD_LIBRARY_PATH is necessary for Python executable to find the libpython3.7m.so.1.0 (as we use non-standard prefix)
ENV LD_LIBRARY_PATH=/python-dist/lib:${LD_LIBRARY_PATH}
RUN wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tar.xz  &&\
    tar xf Python-3.7.0.tar.xz  &&\
    cd Python-3.7.0 &&\
    ./configure --prefix /python-dist --enable-shared  &&\
    make -j 16  &&\
    make install  &&\
    /python-dist/bin/python3.7 -mpip install --upgrade pip &&\
    /python-dist/bin/python3.7 -mpip install numpy matplotlib seaborn sklearn &&\
    cd .. &&\
    rm -rf Python-3.7.0.tar.xz Python-3.7.0

# install Boost
# running initial download separately, as it takes some time
ENV CPATH=/python-dist/include/python3.7m
RUN wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz &&\
    tar -xzf boost_1_68_0.tar.gz &&\
    source /opt/rh/devtoolset-7/enable &&\
    cd boost_1_68_0 &&\
    ./bootstrap.sh --prefix=/usr/local &&\
    ./b2 -j 8 install &&\
    cd .. &&\
    rm -rf boost_1_68_0.tar.gz boost_1_68_0

# install CMake - necessary for building Dataframes and several its dependencies
RUN wget https://cmake.org/files/v3.12/cmake-3.12.3-Linux-x86_64.sh &&\
    chmod +x cmake-3.12.3-Linux-x86_64.sh &&\
    ./cmake-3.12.3-Linux-x86_64.sh --skip-license --prefix=/usr

# XLNT
RUN wget https://github.com/tfussell/xlnt/archive/v1.3.0.zip && unzip v1.3.0.zip && cd xlnt-1.3.0 && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo . && make -j 8 && make install && make clean && cd .. && rm -rf xlnt-1.3.0 v1.3.0.zip

# APACHE ARROW
RUN wget https://github.com/apache/arrow/archive/apache-arrow-0.10.0.zip && unzip apache-arrow-0.10.0.zip && cd arrow-apache-arrow-0.10.0 && mkdir build && cd build && cmake ../cpp -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && make clean && cd ../../ && rm -rf arrow-apache-arrow-0.10.0 apache-arrow-0.10.0.zip

# DATE
# requires libcurl
RUN wget https://github.com/HowardHinnant/date/archive/v2.4.1.zip && unzip v2.4.1.zip && cd date-2.4.1 && mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && make clean && cd ../../ && rm -rf date-2.4.1 v2.4.1.zip

# PYBIND11
RUN wget https://github.com/pybind/pybind11/archive/v2.2.4.zip && unzip v2.2.4.zip && cd pybind11-2.2.4 && mkdir build && cd build && cmake ../ -DPYBIND11_TEST=OFF -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && cd ../../ && rm -rf pybind11-2.2.4 v2.2.4.zip

# {FMT}
RUN wget https://github.com/fmtlib/fmt/archive/5.2.1.zip && unzip 5.2.1.zip && cd fmt-5.2.1 && mkdir build && cd build && cmake ../ -DFMT_TEST=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && cd ../../ && rm -rf fmt-5.2.1 5.2.1.zip

# rapidjson
RUN wget https://github.com/Tencent/rapidjson/archive/v1.1.0.zip && unzip v1.1.0.zip && cd rapidjson-1.1.0 && mkdir build && cd build && cmake ../ -DRAPIDJSON_BUILD_TESTS=OFF -DRAPIDJSON_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && cd ../../ && rm -rf rapidjson-1.1.0 v1.1.0.zip

# install patchelf utility (used in packaging)
RUN wget https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz &&\
    tar xf patchelf-0.9.tar.gz &&\
    cd patchelf-0.9 &&\
    ./configure &&\
    make install &&\
    cd .. &&\
    rm -rf patchelf-0.9 patchelf-0.9.tar.gz

RUN wget https://github.com/commercialhaskell/stack/releases/download/v1.9.1/stack-1.9.1-linux-x86_64.tar.gz &&\
    tar xf stack-1.9.1-linux-x86_64.tar.gz &&\
    cp stack-1.9.1-linux-x86_64/stack /usr/local/bin/stack &&\
    rm -rf stack-1.9.1-linux-x86_64.tar.gz stack-1.9.1-linux-x86_64 &&\
    yum install -y gmp-devel &&\
    stack setup 8.6.2 &&\
    stack --resolver nightly-2018-11-23 build Cabal MissingH typed-process

RUN wget https://www.mirrorservice.org/sites/dl.fedoraproject.org/pub/epel/7/x86_64/Packages/p/p7zip-16.02-10.el7.x86_64.rpm &&\
    rpm -U --quiet p7zip-16.02-10.el7.x86_64.rpm &&\
    rm p7zip-16.02-10.el7.x86_64.rpm
