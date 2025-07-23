"""
our data is a list of trajectories. Each trajectory is a series of points with the following format:
tid,label,lat,lon,day,hour,category
where:
- tid: trajectory id (same for the whole trajectory)
- label: user id (same for the whole trajectory and user)
- lat: latitude
- lon: longitude
- day: day of the week (0-6)
- hour: hour of the day (0-23)
- category: venue category (e.g. restaurant, park, etc.), in other words, semantic information about the point's location


example:
3845,138,40.7669199741308,-73.9575165692434,0,11,5
3845,138,40.7669199741308,-73.9575165692434,1,14,5
3845,138,40.7669199741308,-73.9575165692434,2,10,5
3845,138,40.7149642107861,-73.9665055274963,3,10,7
3845,138,40.7686066202744,-73.9556396110218,3,22,8
3845,138,40.766606181603706,-73.9571606947745,4,12,5
3845,138,40.766606181603706,-73.9571606947745,5,12,5
3845,138,40.7197747224541,-74.0067127508062,6,0,0
3845,138,40.766606181603706,-73.9571606947745,6,13,5
3845,138,40.722297999999995,-73.988823,6,22,7
3848,138,40.766606181603706,-73.9571606947745,0,11,5
3848,138,40.766606181603706,-73.9571606947745,1,14,5
3848,138,40.766606181603706,-73.9571606947745,2,9,5
3848,138,40.7667291004899,-73.9568603038788,2,12,0
3848,138,40.717776,-73.957848,2,20,7
3848,138,40.766606181603706,-73.9571606947745,4,10,5
3848,138,40.757771000000005,-73.932086,4,20,0
3848,138,40.7263535916991,-74.0019555280681,5,20,0
3848,138,40.724622472400895,-73.99586106037441,5,20,8
3848,138,40.7225464012878,-73.9861336130903,5,22,8
3848,138,40.7230298063774,-73.9889399892055,5,23,8
3848,138,40.737831,-73.981022,6,1,8
3852,138,40.7669055013556,-73.9578494834753,0,10,5
3852,138,40.72945076647871,-74.00129559036391,0,20,8
3852,138,40.737831,-73.981022,1,0,8
3852,138,40.7459817610903,-73.9536162640663,1,3,2
3852,138,40.7669055013556,-73.9578494834753,2,9,5
3852,138,40.768369,-73.955709,2,18,8
3852,138,40.7459817610903,-73.9536162640663,2,23,2
3852,138,40.7669055013556,-73.9578494834753,3,18,5
3852,138,40.768369,-73.955709,3,18,8
3852,138,40.7459817610903,-73.9536162640663,4,10,2
3852,138,40.7669055013556,-73.9578494834753,5,15,5
3852,138,40.764154619063,-73.9868969055432,5,22,8
3852,138,40.7635128499675,-73.9869198616602,6,1,8
3852,138,40.7459817610903,-73.9536162640663,6,10,2

we will transform this data into a slightly different format.
instead of tid, label, lat, lon, day, hour, category
each point will have tid, lat, lon, timestamp
where timestamp is in a standardized format

"""

if __name__ == '__main__':
    import argparse
    import pandas as pd
    from datetime import datetime, timedelta

    # get filename from argparse arg
    parser = argparse.ArgumentParser(description="Convert CSV to DeepMove format")
    parser.add_argument('--file',  type=str, default='../data/train.csv', help='input CSV file')

    args = parser.parse_args()
    input_file = args.file
    data = pd.read_csv(input_file)

    # drop label, category
    data = data.drop(columns=['label', 'category'])

    # convert day and hour to timestamp. Use a fixed starting date for simplicity
    start_date = datetime(2025, 6, 30)
    data['timestamp'] = data.apply(
        lambda row: (start_date + timedelta(days=row['day'], hours=row['hour'])).strftime('%Y-%m-%d %H:%M:%S'),
        axis=1
    )
    # drop day and hour
    data = data.drop(columns=['day', 'hour'])
    # save to the same csv file
    data.to_csv(input_file, index=False)