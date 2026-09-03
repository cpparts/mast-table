from ipyvuetify import VuetifyTemplate
import traitlets
from typing import Any, Dict, List, cast


class Select(VuetifyTemplate):
    template_file = __file__, "cross_filter_select.vue"

    value = traitlets.Any().tag(sync=True)
    label = traitlets.Unicode().tag(sync=True)
    clearable = traitlets.Bool().tag(sync=True)
    return_object = traitlets.Bool().tag(sync=True)
    items = traitlets.List(cast(List[Dict[str, Any]], [])).tag(sync=True)
    filtered = traitlets.Bool().tag(sync=True)
    count = traitlets.Int().tag(sync=True)
    multiple = traitlets.Bool().tag(sync=True)
    messages = traitlets.Unicode().tag(sync=True)
    search = traitlets.Any().tag(sync=True)
