<template>
  <v-container fluid class="base-mast-table">
    <div class="base-mast-table-toolbar">
      <v-btn
        :color="filter_tray_open ? '#b4dbe9' : undefined"
        @click="filter_tray_open = !filter_tray_open"
      >
        {{ filter_tray_open ? 'Hide Conditions' : 'Show Conditions' }}
        <v-icon>
          {{ filter_tray_open ? 'mdi-chevron-left' : 'mdi-chevron-right' }}
        </v-icon>
      </v-btn>

      <div v-if="show_if_empty || items.length">
        <v-col style="max-width: 400px;">
          <v-select
            v-model="headers_visible"
            :items="headers_avail"
            item-title="title"
            item-value="key"
            label="Display columns"
            multiple
            density="compact"
            hide-details
            class="column-select"
            bg-color="transparent"
            :menu-props="{ contentClass: 'base-mast-column-menu' }"
          >
            <template v-slot:selection="{ index }">
              <span v-if="index === 0">
                ({{ headers_visible.length }} of {{ headers_avail.length }} columns displayed)
              </span>
            </template>
            <template v-slot:prepend-item>
              <v-list-item
                class="select-all-item"
                density="compact"
                ripple
                @mousedown.prevent
                @click="
                  headers_visible.length < headers_avail.length
                    ? headers_visible = [...headers_avail]
                    : headers_visible = []
                "
              >
                <div class="select-all-content">
                  <v-icon size="24">
                    {{ headers_visible.length == headers_avail.length ? 'mdi-close-box' : headers_visible.length ? 'mdi-minus-box' : 'mdi-checkbox-blank-outline' }}
                  </v-icon>
                  <span>
                    {{ headers_visible.length < headers_avail.length ? "Select All" : "Clear All" }}
                  </span>
                </div>
              </v-list-item>
              <v-divider></v-divider>
            </template>
          </v-select>
        </v-col>
      </div>

      <v-menu v-model="menu_open" :close-on-content-click="false">
        <template v-slot:activator="{ props }">
          <v-btn
            v-bind="props"
            icon="mdi-menu"
            variant="text"
          />
        </template>

        <v-card class="base-mast-settings-menu" min-width="300">
          <v-switch
            v-model="show_tooltips"
            label="Show column definition on hover"
            color="#00617e"
            hide-details
          />
        </v-card>
      </v-menu>

      <div v-if="mission == 'list_products' && enable_load_in_app">
        <v-tooltip location="top">
        <template v-slot:activator="{ props }">
          <v-btn
            v-bind="props"
            :disabled="selected_rows.length === 0"
            class="open-in elevation-1"
            @click="open_selected_rows_in_aladin"
            ><v-icon>mdi-open-in-app</v-icon>aladin</v-btn>
          </template>
          <div style="text-align: center;">Download, open selection<br />in mast-aladin-lite</div>
        </v-tooltip>

        <v-tooltip location="top">
        <template v-slot:activator="{ props }">
          <v-btn
            v-bind="props"
            :disabled="selected_rows.length === 0"
            class="open-in elevation-1"
            @click="open_selected_rows_in_jdaviz"
            ><v-icon>mdi-open-in-app</v-icon>jdaviz</v-btn>
          </template>
          <div style="text-align: center;">Download, open<br />selection in jdaviz</div>
        </v-tooltip>
      </div>
    </div>

    <v-data-table-server
      :headers="headers.filter(h => headers_visible.includes(h.key))"
      :items="items"
      :items-length="server_items_length"
      :item-value="item_key"
      v-model:items-per-page="items_per_page"
      v-model:sort-by="sort_by"
      :items-per-page-options="[
        { value: 5, title: '5' },
        { value: 10, title: '10' },
        { value: 15, title: '15' },
        { value: -1, title: 'All' }
      ]"
      :show-select="show_rowselect"
      :footer-props="{ 'itemsPerPageText': 'Items per page:' }"
      v-model="selected_rows"
      v-model:options="table_options"
      class="base-mast-data-table elevation-2"
      density="compact"
    >
      <template
        v-for="header in headers.filter(h => headers_visible.includes(h.key))"
        v-slot:[`header.${header.key}`]="{
          column,
          toggleSort,
          getSortIcon,
          isSorted
        }"
      >
        <div
          class="mast-header-cell"
          :class="{ 'mast-header-cell--sortable': column.sortable !== false }"
          @click="toggleSort(column)"
        >
          <v-tooltip
            v-if="show_tooltips && header.description"
            location="top"
          >
            <template v-slot:activator="{ props }">
              <span v-bind="props" class="mast-header-title">
                <strong>{{ column.title }}</strong>
              </span>
            </template>

            <div style="max-width: 300px">
              <strong>{{ column.title }}</strong>: {{ header.description }}
            </div>
          </v-tooltip>

          <span v-else class="mast-header-title">
            <strong>{{ column.title }}</strong>
          </span>

          <v-icon
            v-if="column.sortable !== false"
            class="mast-header-sort"
            :class="{ 'mast-header-sort--active': isSorted(column) }"
            size="16"
          >
            {{ getSortIcon(column) }}
          </v-icon>
        </div>
      </template>
    </v-data-table-server>
  </v-container>
</template>


<style scoped>
:root {
  color-scheme: light dark;
}

.base-mast-table-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.column-select {
  min-width: 300px;
  max-width: 400px;
}

/* header */
:deep(.base-mast-data-table thead) {
  background-color: light-dark(#b4dbe9, #013b4d) !important;
}
:deep(.base-mast-data-table thead th) {
  background-color: light-dark(#b4dbe9, #013b4d) !important;
  color: light-dark(black, white) !important;
}

/* sortable column icons */
.mast-header-cell {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
}
.mast-header-title {
  display: inline-flex;
  align-items: center;
}
.mast-header-sort {
  opacity: 0;
  transition: opacity 0.15s;
}
.mast-header-cell:hover .mast-header-sort {
  opacity: 0.38;
}
.mast-header-sort--active {
  opacity: 1;
}

/* alternating rows */
:deep(.base-mast-data-table .v-data-table__tr:nth-child(even)) {
  background: #f1f2f7 !important;
}
:deep(.base-mast-data-table .v-data-table__tr:nth-child(odd)) {
  background: white !important;
}

/* selected */
:deep(.base-mast-data-table tbody tr:has(.v-selection-control--dirty)) td {
  background-color: #bdf0fd !important;
  color: black !important;
}

/* footer */
:deep(.base-mast-data-table .v-data-table-footer) {
  background-color: light-dark(#b4dbe9, #013b4d) !important;
  color: black !important;
}
:deep(.base-mast-data-table .v-data-table-footer .v-field) {
  background-color: light-dark(#b4dbe9, #013b4d) !important;
}
:deep(.base-mast-data-table .v-data-table-footer .v-field__input),
:deep(.base-mast-data-table .v-data-table-footer .v-label) {
  color: black !important;
}
:deep(.base-mast-data-table .v-data-table-footer .v-btn) {
  color: black !important;
}
:deep(.base-mast-data-table .v-data-table-footer .items-per-page-options) {
  color: black !important;
}

/* reducing whitespace between rows */
/* compact body rows */
:deep(.base-mast-data-table tbody .v-data-table__td) {
  white-space: nowrap !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  height: 24px !important;
  line-height: 1 !important;
}
/* make selection control fit the row */
:deep(.base-mast-data-table tbody .v-selection-control) {
  min-height: 24px !important;
  height: 24px !important;
}
:deep(.base-mast-data-table tbody .v-selection-control__wrapper) {
  height: 24px !important;
  width: 24px !important;
}
/* don't let checkbox's internal input create extra space */
:deep(.base-mast-data-table tbody .v-selection-control__input) {
  height: 24px !important;
  width: 24px !important;
}

/* reducing footer whitespace */
/* compact table footer */
:deep(.base-mast-data-table .v-data-table-footer) {
  min-height: 45px !important;
  height: 45px !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
/* compact items-per-page select */
:deep(.base-mast-data-table .v-data-table-footer .v-field) {
  min-height: 28px !important;
  height: 28px !important;
}
:deep(.base-mast-data-table .v-data-table-footer .v-field__input) {
  min-height: 28px !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
:deep(.base-mast-data-table .v-data-table-footer .v-select) {
  height: 28px !important;
}

/* select/clear columns */
.select-all-item {
  min-height: 36px !important;
  padding: 0 20px !important;
}

</style>

<style>
.base-mast-column-menu .v-list {
  padding: 4px 0 !important;
}

.base-mast-column-menu .v-list-item {
  min-height: 32px !important;
  height: 32px !important;
  padding: 0 12px !important;
}

.base-mast-column-menu .v-list-item__content {
  padding: 0 !important;
}

.base-mast-column-menu .v-list-item__prepend {
  padding: 0 !important;
}

.base-mast-column-menu .v-list-item__append {
  padding: 0 !important;
}

.base-mast-column-menu .v-list-item__spacer {
  width: 8px !important;
}

.base-mast-column-menu .select-all-item {
  min-height: 32px !important;
  height: 32px !important;
  padding: 0 20px !important;
}

.base-mast-column-menu .select-all-content {
  display: flex;
  align-items: center;
  gap: 16px;
  width: 100%;
  white-space: nowrap;
}

/* open-in buttons */
.open-in {
  background-color: light-dark(#b4dbe8, #00617e) !important;
  margin-left: 0.5em !important;
  width: 7em;
}
.open-in:not(:disabled):hover {
  background-color: light-dark(#b4dbe8, #00617e) !important;
  color: light-dark(black, white) !important;
}
.open-in:disabled {
  background-color: light-dark(#e0e0e0, #424242) !important;
  color: light-dark(black, white) !important;
  opacity: 0.6 !important;
}

/* defintions toggle menu */
.base-mast-settings-menu {
  padding: 8px 12px !important;
  overflow: visible !important;
}
.base-mast-settings-menu .v-switch {
  overflow: visible !important;
}
</style>