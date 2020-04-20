const merge = require("webpack-merge");
const UglifyJsPlugin = require("uglifyjs-webpack-plugin");
const OptimizeCSSAssetsPlugin = require("optimize-css-assets-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");

const common = require("./webpack.dev.js");

module.exports = merge(common, {
  mode: "production",

  optimization: {
    minimizer: [
      new UglifyJsPlugin({
        cache: true,
        parallel: true,
        sourceMap: true
      }),

      //new MiniCssExtractPlugin({
      //  filename: "[name].[hash:5].css",
      //  chunkFilename: "[id].[hash:5].css"
      //}),

      new OptimizeCSSAssetsPlugin({})
    ]
  },

  watch: false
});
