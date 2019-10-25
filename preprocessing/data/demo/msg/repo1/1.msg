
    Support setting client name for JedisCluster 1.3.4 (#1386)
    [Motivation]
    JedisCluster should support setting client name similar with other types
    of Jedis.
    [Modifications]
    (1) add private member: clientName for JedisClusterInfoCache;
    (2) add new constructor with clientName for JedisCluster.
    [Result]
    JedisCluster support setting client name.
