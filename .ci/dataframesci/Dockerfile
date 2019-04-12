FROM ubuntu:18.04 AS DataframesCI

RUN apt-get update
RUN apt-get install -y cmake g++ g++-8 git libboost-all-dev libcurl4-openssl-dev libfmt-dev python3-pip rapidjson-dev unzip wget
RUN pip3 install numpy matplotlib seaborn sklearn

# XLNT
RUN wget https://github.com/tfussell/xlnt/archive/v1.3.0.zip && unzip v1.3.0.zip && cd xlnt-1.3.0 && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo . && make -j 8 && make install && make clean && cd .. && rm -rf xlnt-1.3.0 xlnt-1.3.0.zip

# APACHE ARROW
RUN wget https://github.com/apache/arrow/archive/apache-arrow-0.10.0.zip && unzip apache-arrow-0.10.0.zip && cd arrow-apache-arrow-0.10.0 && mkdir build && cd build && cmake ../cpp -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && make clean && cd ../../ && rm -rf arrow-apache-arrow-0.10.0 apache-arrow-0.10.0.zip

# DATE
RUN wget https://github.com/HowardHinnant/date/archive/v2.4.1.zip && unzip v2.4.1.zip && cd date-2.4.1 && mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && make clean && cd ../../ && rm -rf date-2.4.1 v2.4.1.zip

# PYBIND11
RUN wget https://github.com/pybind/pybind11/archive/v2.2.4.zip && unzip v2.2.4.zip && cd pybind11-2.2.4 && mkdir build && cd build && cmake ../ -DPYBIND11_TEST=OFF -DCMAKE_BUILD_TYPE=Release && make -j 8 && make install && cd ../../ && rm -rf pybind11-2.2.4 v2.2.4.zip

# RUN git clone https://github.com/luna/Dataframes.git
# RUN mkdir build
# WORKDIR build
# RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../Dataframes/native_libs/src
# RUN make -j 8
# WORKDIR ../Dataframes
# RUN ../build/DataframeHelperTests -p -r detailed
