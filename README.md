Solution for 22nd place (only Prophet based).

In summary: 2 separate models for segment1 and segment 2 - both at country level.

#### Segment 1 params:
```
{
    'holiday': True,
    'holidays_prior_scale': 0.1,
    'changepoint_prior_scale': .6,
    'seasonal': {
        'weekly': (7, 5, 0.7, 'additive'),
    }
}
```
#### Segment 2 params:

```
{
    'holiday': True,
    'holidays_prior_scale': 0.01,
    'changepoint_prior_scale': 0.03,
    'seasonal': {
        'monthly': (30, 1, 0.1, 'additive'),
        'weekly': (7, 1, 0.55, 'additive'),
    }
}
```

National holidays were used in both the models (`scrape_national_holidays` fn in `utils.py` scrapes and return a dataframe for holidays) but it was significantly more effective for segment1. I used last 30, 60 and 90 days for validation. Best average valation for segment 1 was 9.08 and for segment 2 it was around 18. Log transformation used for segment 2 only.

Considering the series after `2018-04-01` produced best result but I forgot to truncate the series while making the submission.