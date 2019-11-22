def scale_hourly_shape_profile_by_monthly_total(hourly_profile,monthly_total,leap_year=True):
    """Scale hourly shape profile based on monthly total profile
    :param list hourly_profile: list of hourly profile
    :param list monthly_total: list of monthly total
    :param bool leap_year: indicates the query year is a leap year or not
    :return: (*list*) -- scaled_hourly_profile  
    """
    if leap_year:
        days_per_month = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        
    hour_num = [day*24 for day in days_per_month]
    hour_month_mapping = []
    month_factor = [0]*12
    hour_count = 0
    
    for i,hour in enumerate(hour_num):
        hour_month_mapping += [i]*hour
        month_sum = sum(hourly_profile[hour_count:hour_count+hour])
        hour_count += hour
        month_factor[i] = monthly_total[i]/month_sum
    scaled_hourly_profile = [d*month_factor[hour_month_mapping[i]] for i,d in enumerate(hourly_profile)] 
    
    return scaled_hourly_profile