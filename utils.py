'''
get ip scope from ip category
'''
def get_ip_scope(ip_cat):
    if ip_cat == "INTERNET" or ip_cat == "MULTICAST" or ip_cat == "CURR_NET":
        return "Internet"
    if ip_cat == "BENCH" or "PRIV" in ip_cat:
        return "Private network"
    if ip_cat == "LINK-LOCAL" or ip_cat == "BROADCAST":
        return "Subnet"
    if ip_cat == "LOOPBACK":
        return "Host"
    return ""

'''
get ip zone from ip address
'''
def get_ip_zone(ip_address, zone_id):
    return ip_address.split(".")[zone_id - 1]

'''
convert timestamp in seconds to hours
'''
def get_duration(duration):
    return duration // 3600

'''
get end time of the event
'''
def get_end_time(start_hour, start_minute, start_second, duration, type):
    tmp_duration, second_duration = divmod(duration, 60)
    hour_duration, minute_duration = divmod(tmp_duration, 60)

    end_second = start_second + second_duration
    end_minute = start_minute + minute_duration
    end_hour = start_hour + hour_duration

    if end_second >= 60:
        end_second = end_second - 60
        end_minute = end_minute + 1
    if end_minute >= 60:
        end_minute = end_minute - 60
        end_hour = end_hour + 1
    if end_hour >= 24:
        end_hour = end_hour - 24

    if type == "hour":
        return end_hour
    if type == "minute":
        return end_minute
    if type == "second":
        return end_second

'''
get sum of scores
'''
def get_sum(scores):
    return sum(scores)

'''
get concatenat values 
'''
def concatenate_values(values):
    str_values = [str(value) for value in values]
    return "-".join(str_values)

'''
get ratio
'''
def get_ratio(numerator, denominator):
    return float(numerator) / (denominator + 1)