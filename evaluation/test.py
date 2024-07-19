from datetime import datetime, timedelta
import random
from evaluation.eval_framework import remove_duplicates, unpack_tuples

def generate_random_dates(year):
    """
    Generate a list of tuples containing two random dates from each week in the given year.

    Parameters:
    year (int): The year for which to generate the random dates.

    Returns:
    list: A list of tuples, each containing two random dates from the same week.
    """
    random_dates = []
    start_date = datetime(year, 1, 1)
    #end_date = datetime.now() - timedelta(days=1)  # Last possible day is yesterday's date

    # Find the first Monday of the year
    while start_date.weekday() != 0:
        start_date += timedelta(days=1)

    while start_date.year == year:
        # Calculate the end date of the current week
        end_date = start_date + timedelta(days=6)
        #if week_end_date > end_date:
        #    week_end_date = end_date

        # Generate two random dates within the current week
        random_date1 = start_date + timedelta(days=random.randint(0, 6))
        random_date2 = start_date + timedelta(days=random.randint(0, 6))

        # Ensure the dates are within the same week
        if random_date1.weekday() > random_date2.weekday():
            random_date1, random_date2 = random_date2, random_date1

        random_dates.append((random_date1, random_date2))

        # Move to the next week
        start_date += timedelta(days=7)
    
    random_dates = remove_duplicates(
        unpack_tuples(
            generate_random_dates(year)
            )
        )

    threshold_date = datetime.now() - timedelta(days=1)
    random_dates = [date for date in date_list if date > threshold_date]

    return random_dates