"""For sites that expose one dated sitemap file per day (e.g. abplive.com's
news-DD-MM-YYYY.xml), generate the list of daily sitemap URLs to feed to discover_urls
— each daily file is itself a plain <urlset>, so no special parsing is needed beyond that."""

from datetime import datetime, timedelta
from typing import Iterable


def daily_sitemap_urls(template: str, start_date: str, end_date: str) -> Iterable[str]:
    """template uses {dd}, {mm}, {yyyy} placeholders. Dates are 'YYYY-MM-DD'. Yields most
    recent day first, since that's the content most likely to still be uncollected."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    day = end
    while day >= start:
        yield template.format(dd=f"{day.day:02d}", mm=f"{day.month:02d}", yyyy=day.year)
        day -= timedelta(days=1)
