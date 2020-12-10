#!/usr/bin/env bash

function wait_for_cassandra() {
    count=0
    #
    # The easiest way to wait for Cassandra would be to use:
    #   cqlsh -e "describe cluster;"
    # However, since Cassandra is installed as a deb package, it means that the provided python
    # modules which are used by the command line tool 'cqlsh' are installed inside 'dist-packages'.
    # 'cqlsh' is a wrapper that might pick up a python interpreter that does not load packages
    # from that directory, so we could update the PYTHONPATH environmental variable here just for
    # this script. However doing this would risk mixing modules of different python versions
    # together, with potentially unpredictable effects (though limited to this script).
    #
    # To preserve our future mental sanity we run a small python program that does not depend
    # on libraries installed by cassandra, but just packages we already installed (the driver).
    cmd="from cassandra import cluster; c = cluster.Cluster([\"127.0.0.1\"]); c.connect()"
    while ! python -c "${cmd}" > /dev/null 2>&1; do
        echo "waiting for cassandra"
        if [ $count -gt 60 ]
        then
            exit
        fi
        (( count += 1 ))
        sleep 1
    done
    echo "cassandra is ready"
}

wait_for_cassandra
