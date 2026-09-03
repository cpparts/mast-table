<template>
  <input
    type="number"
    :value="value"
    :min="min"
    :max="max"
    :step="step"
    @change="onChange"
    class="crossfilter-input"
  />
</template>

<script>
export default {
  methods: {
    onChange(event) {
      let newValue = Number(event.target.value)
      if (Number.isNaN(newValue)) {return}
      newValue = Math.max(this.min, Math.min(this.max, newValue))
      this.value = newValue
    },
  },
}
</script>

<style>
.crossfilter-input {
    width: 100px;
    height: 32px;
    padding: 4px 8px;
    font-size: 16px;
    line-height: 20px;
}
.crossfilter-input::-webkit-inner-spin-button, 
.crossfilter-input::-webkit-outer-spin-button {
  -webkit-appearance: auto;
  opacity: 1;
  margin: 0;
}
.crossfilter-input:hover,
.crossfilter-input:focus,
.crossfilter-input:focus-visible,
.crossfilter-input:active {
    outline: none !important;
    border-color: none !important;
    box-shadow: none !important;
}
</style>