BASE_DIR = '/home/bishwarup/av_ltfs2'
BASE_URL = 'https://www.officeholidays.com/countries/india'

holiday_window = {
    'Republic Day': (0, 0),
    'Maha Shivaratri/Shivaratri': (-1, 1),
    'Holi': (-1, 1),
    'Rama Navami': (-1, 1),
    'Mahavir Jayanti': (0, 0),
    'Good Friday': (0, 0),
    'Buddha Purnima/Vesak': (0, 0),
    "Idu'l Fitr": (-2, 1),
    "Independence Day": (-1, 1),
    "Id-ul-Zuha(Bakrid)": (-1, 1),
    "Janmashtarni (Vaishnav)": (-1, 0),
    "Muharram/Ashura": (-2, 2),
    'Mahatma Gandhi Jayanti': (0, 0),
    "Dussehra": (-10, 0),
    "Diwali (Deepavali)": (-2, 1),
    "'Milad-un-Nabi or Id-e- Milad (birthday of Prophet Mohammad)": (0, 0),
    "Guru Nanak's Birthday": (0, 0),
    "Christmas Day":(-1, 1)
    
}
statewise_params = {
    "WEST BENGAL": {
        "holiday": True,
        "changepoint_prior_scale": 0.9,
        "holidays_prior_scale": 0.05,
        "seasonal": {
            "quarterly": (91, 25, 0.55, None),
            "monthly": (29.5, 21, 0.55, None),
            "weekly": (7, 7, 0.9, None)
        }
    },
    "CHHATTISGARH": {
        "holiday": False,
        "changepoint_prior_scale": 0.2,
        "holidays_prior_scale": 0.05,
        "seasonal": {
            "yearly": (365, 2, 0.05, None),
            "quarterly": (91, 30, 0.55, None),
            "monthly": (30, 35, 0.55, None),
        }    
    },
    "ASSAM": {
        "holiday": True,
        "changepoint_prior_scale": 1.,
        "holidays_prior_scale": 0.05,
        "seasonal": {
            "yearly": (365, 1, 0.05, None),
            "quarterly": (91, 53, 0.55, None),
            "monthly": (30.5, 10, 0.55, None),
            "weekly": (6.5, 2, 0.05, None)
        }
    },
    "KARNATAKA": {
        "holiday": False,
        "changepoint_prior_scale": 1.,
        "holidays_prior_scale": 10,
        "seasonal": {
            "yearly": (365, 1, 0.05, None),
            "quarterly": (91, 45, 0.9, None),
            "monthly": (30.5, 31, 0.55, None),
            "weekly": (7, 1, 0.05, None)
        }
    },
    "MAHARASHTRA": {
        "holiday": True,
        "changepoint_prior_scale": 0.05,
        "holidays_prior_scale": 0.01,
        "seasonal": {
            "yearly": (365, 2, 1., None),
            "quarterly": (91, 7, 1., None),
            "monthly": (30.5, 5, 0.05, None),
        }
    },
    "MADHYA PRADESH": {
        "holiday": True,
        "changepoint_prior_scale": 3,
        "holidays_prior_scale": 0.25,
        "seasonal": {
            "yearly": (365, 3, 0.9, None),
            "quarterly": (91, 32, 0.35, None),
            "monthly": (30.5, 50, 0.75, None),
            "weekly": (7, 4, 0.95, None)
        },
    },
    "GUJARAT": {
        "holiday": False,
        "changepoint_prior_scale": 0.05,
        "holidays_prior_scale": 10,
        "seasonal": {
            "quarterly": (91, 51, 1., None),
            "weekly": (7, 3, 0.1, None)
        }
    },
    "KERALA": {
        "holiday": True,
        "changepoint_prior_scale": 0.05,
        "holidays_prior_scale": 15,
        "seasonal": {
        "yearly": (364.5, 15, 0.05, None),
            "quarterly": (91, 40, 10., None),
            "monthly": (30.5, 35, 0.05, None),
        }
    },
    "ORISSA": {
        "holiday": False,
        "changepoint_prior_scale": 0.05,
        "holidays_prior_scale": 0.05,
        "seasonal": {
            "yearly": (365, 1, 0.01, None),
            "quarterly": (91, 30, 0.05, None),
            "monthly": (30.5, 1, 0.05, None),
            "weekly": (7, 15, 0.05, None)
        }
    },
    "TAMIL NADU": {
        "holiday": True,
        "changepoint_prior_scale": 2,
        "holidays_prior_scale": 5,
        "seasonal": {
            "quarterly": (91.05, 46, 3, None),
            "monthly": (30.5, 32, 0.05, None),
            "weekly": (7, 2, 0.05, None)
        }
    },
    "UTTAR PRADESH": {
        "holiday": True,
        "changepoint_prior_scale": 5,
        "holidays_prior_scale": 10,
        "seasonal": {
            "yearly": (365, 1, 0.55, None),
            "quarterly": (91.1, 45, 10, None),
            "monthly": (30.5, 5, 2, None),

        }
    },
    "TRIPURA": {
        "holiday": False,
        "changepoint_prior_scale": 3.2,
        "holidays_prior_scale": 0.2,
        "seasonal": {
            "yearly": (365, 1, 0.55, None),
            "quarterly": (91.1, 30, 0.55, None),
            "monthly": (29.7, 5, 0.05, None),
        }
    },
    "JHARKHAND": {
        "holiday": True,
        "changepoint_prior_scale": 10,
        "holidays_prior_scale": 0.1,
        "seasonal": {
            "yearly": (365, 1, 0.05, None),
            "quarterly": (91.2, 40, 1.55, None),
            "monthly": (30.7, 5, 1, None),
            "weekly": (7, 1, 0.2, None)
        }
    }
}


aggregated_params = {
    
    2: {
        'holiday': True,
        'holidays_prior_scale': 0.01,
        'changepoint_prior_scale': 0.03,
        'seasonal': {
            'monthly': (30, 1, 0.1, 'additive'),
            'weekly': (7, 1, 0.55, 'additive'),
        }
    }   
}