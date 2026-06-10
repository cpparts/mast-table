<template>
  <v-row class="mast-table">
    <v-col cols="auto">
      <v-btn class="{'conditions-active': filter_tray_open}" @click="filter_tray_open = !filter_tray_open" :color="filter_tray_open ? '#b4dbe9' : undefined">
        Conditions
        <v-icon right>mdi-filter</v-icon>
      </v-btn>
    </v-col>
  <div v-if="show_if_empty || items.length">
    <v-col style="max-width: 400px;">
      <div class="row-select">
        <v-select
          class="no-hint"
          v-model="headers_visible"
          :items="headers_avail"
          @change="$emit('update:headers_visible', $event)"
          label="Display columns"
          multiple
          dense
        >
        <template v-slot:selection="{ item, index }">
          <span
            v-if="index === 0"
            class="grey--text text-caption"
          >
            ({{ headers_visible.length}} of {{headers_avail.length}} columns displayed)
          </span>
        </template>
        <template v-slot:prepend-item>
          <v-list-item
            class="elevation-1"
            ripple
            @mousedown.prevent
            @click="() => {if (headers_visible.length < headers_avail.length) { headers_visible = headers_avail} else {headers_visible = []}}"
          >
            <v-list-item-action>
              <v-icon>
                {{ headers_visible.length == headers_avail.length ? 'mdi-close-box' : headers_visible.length ? 'mdi-minus-box' : 'mdi-checkbox-blank-outline' }}
              </v-icon>
            </v-list-item-action>
            <v-list-item-content>
              <v-list-item-title>
                {{ headers_visible.length < headers_avail.length ? "Select All" : "Clear All" }}
              </v-list-item-title>
            </v-list-item-content>
          </v-list-item>
          <v-divider class="mt-2"></v-divider>
        </template>
      </v-select>
      </div>
      </v-col>
    </div>
  <div class="toolbar-icons">
    <jupyter-widget :widget="popout_button"></jupyter-widget>

    <v-menu v-model="menu_open" :close-on-content-click="false" anchor="start end">
      <template v-slot:activator="{ on }">
        <v-btn v-on="on" icon><v-icon>mdi-menu</v-icon></v-btn>
      </template>

      <v-card min-width="300">
        <v-list>
          <v-list-item>
            <v-switch v-model="show_tooltips" color="#00617e" label="Show column definition on hover"></v-switch>
          </v-list-item>
        </v-list>
      </v-card>
    </v-menu>
  </div>

  <v-col class="d-flex justify-end">
  <template v-if="show_load_buttons">
    <v-tooltip top>
    <template v-slot:activator="{ on }">
      <v-btn
        v-on="on"
        density="compact"
        :disabled="no_product_selected" 
        class="open-in elevation-1"
        @click="open_selected_rows_in_aladin"
        ><v-icon>mdi-open-in-app</v-icon>aladin</v-btn>
      </template>
      <div style="text-align: center;"">Download, open selection<br />in mast-aladin-lite</div>
    </v-tooltip>

    <v-tooltip top>
    <template v-slot:activator="{ on }">
      <v-btn 
        density="compact"
        v-on="on"
        :disabled="no_product_selected" 
        class="open-in elevation-1"
        @click="open_selected_rows_in_jdaviz"
        ><v-icon>mdi-open-in-app</v-icon>jdaviz</v-btn>
      </template>
      <div style="text-align: center;"">Download, open <br />selection in jdaviz</div>
    </v-tooltip>
</template>
</v-col>

<v-container fluid>
    <v-data-table
      :headers="headers_visible_sorted_description"
      :items="items"
      :item-key="item_key"
      :show-select="show_rowselect"
      :single-select="!multiselect"
      :items-per-page="items_per_page"
      v-model="selected_rows"
      class="elevation-2"
      dense
    >
    <template v-for="h in headers_visible_sorted_description" v-slot:[`header.${h.value}`]="{ header }">
      <template v-if="show_tooltips">
        <v-tooltip top>
          <template v-slot:activator="{ on }">
            <span v-on="on"><strong>{{h.name}}</strong></span>
          </template>
            <div style="max-width: 300px">
              <strong>{{h.name}}</strong>: {{h.description}}
            </div>
        </v-tooltip>
      </template>
      <template v-else>
          <span><strong>{{h.name}}</strong></span>
      </template>
    </template>
    </v-data-table>
</v-container>
</v-row>
</template>

<script>
module.exports = {
  props: ['column_descriptions', 'show_tooltips', 'popout_button', 'filter_tray_open'],
  computed: {
    headers_visible_sorted() {
      return this.headers_avail.filter(item => this.headers_visible.indexOf(item) !== -1);
    },
    headers_visible_sorted_description() {
      return this.headers_visible_sorted.map(item => {
        return {'name': item, 'value': item, 'description': this.get_header_description(item)}
      })
    },
    no_product_selected() {
      return this.selected_rows.length == 0
    },
    show_load_buttons() {
      return this.mission == 'list_products' && this.enable_load_in_app
    },
  },
  methods: {
    get_header_description(item) {
      entry = this.column_descriptions.find(entry => entry.name == item)
      return entry !== undefined ? entry.description : null;
    },
  }
};
</script>


<style scoped>
:root {
  color-scheme: light dark;
}
.v-tooltip__content {
  opacity: 1 !important;
  background-color: light-dark(white, black) !important;
  color: light-dark(black, white) !important;
}
.v-data-table-header, .v-data-footer {
  background-color: light-dark(#b4dbe9, #013b4d) !important;
  .v-data-table-header__icon.mdi {
      hover {
        color: rgba(0, 0, 0, 0.38) !important;
      }
      active {
        color: light-dark(black, white) !important;
      }
  }
  .sortable {
    color: light-dark(black, white) !important;
  }
}
.v-data-table {
    td {
      text-wrap: nowrap !important;
    }
}
.v-btn.open-in {
  background-color: light-dark(#b4dbe8, #00617e) !important;
  margin-left: 0.5em !important;
  width: 7em;
}
.v-btn:hover.open-in {
  background-color: light-dark(#ff9d42, #c75109) !important;
  color: light-dark(black, white) !important;
}
.v-btn:disabled.open-in {
  color: light-dark(black, white) !important;
}
.v-list-item.primary--text.v-list-item--active.v-list-item--link {
  background-color: light-dark(#bcedfc, #bbedfc) !important;
  color: black !important;
}
.v-list-item.primary--text.v-list-item--active.v-list-item--link::before {
  /* prevent low-opacity black overlay on selected items */
  opacity: 0 !important;
}
 .v-icon.notranslate.v-data-table-header__icon {
  color: white !important;
}
.v-data-table__selected {
  background-color: #bdf0fd !important;
  color: black !important;
  font-weight: bold;
}
.v-list-item.primary--text.v-list-item--active {
  opacity: 1 !important;
}
.v-icon.mdi-checkbox-marked {
  color: black !important;
}
.row-select {
  position: relative;
  top: 6px;
  .v-label {
    color: light-dark(black, white) !important;
  }
}
.toolbar-icons {
  display: flex;
  align-items: center;
  gap: 4px;
  position: relative;
  top: -3px;
}
.v-data-table tbody tr:nth-of-type(even) {
    background-color: light-dark(#f1f2f7, black);
}
.v-data-table tbody tr:nth-of-type(odd) {
    background-color: light-dark(white, #172a32);
}
.v-data-table th {
  white-space: nowrap !important;
}
.v-input.v-input--hide-details {
  margin-top: 0.5em !important;
  margin-bottom: 0.5em !important;
}
.v-list-item__action {
  margin-top: 0px !important;
  margin-bottom: 0px !important;
}
v-list-item__content {
  padding-top: 0px !important;
  padding-bottom: 0px !important;
}
</style>