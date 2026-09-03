<template>
  <v-autocomplete
    :items="items"
    v-model="value"
    v-model:search="search"
    item-title="text"
    item-value="value"
    :messages="messages"
    :multiple="multiple"
    :return-object="return_object"
    :clearable="clearable"
    :label="label"
    density="compact"
    class="cross-filter-select"
    :menu-props="{ contentClass: 'cross-filter-menu' }"
    bg-color="white"
    item-color="#00627e"
  >
    <template v-slot:item="{ props, item }">
      <v-list-item v-bind="props">
        <v-list-item-subtitle>
          <v-progress-linear :model-value="count > 0 ? (item.raw.count / count) * 100 : 0"/>
          <div style="display: flex">
            <span v-if="count > 0">{{ ((item.raw.count / count) * 100).toFixed(1) }}%</span>
            <v-spacer />
            <span v-if="filtered">{{ item.raw.count }} of {{ count }} after filtering</span>
            <span v-else>{{ item.raw.count_max }} of {{ count }}</span>
          </div>
        </v-list-item-subtitle>
      </v-list-item>
    </template>
  </v-autocomplete>
</template>

<style>
.cross-filter-selection {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>