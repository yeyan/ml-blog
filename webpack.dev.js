const
  path = require('path'),
  MiniCssExtractPlugin = require('mini-css-extract-plugin'),
  {
    CleanWebpackPlugin
  } = require('clean-webpack-plugin');

module.exports = {
  mode: "development",
  entry: [
    './src/js/app.js',
    './src/scss/app.scss',
  ],
  output: {
    filename: 'js/app.js',
    path: path.resolve(__dirname, 'site/static'),
  },
  module: {
    rules: [
      // js loader
      {
        test: /\.m?js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      },
      // sass loader
      {
        test: /\.s[ac]ss$/,
        use: [
          // style-loader
          MiniCssExtractPlugin.loader,
          // css-loader
          {
            loader: 'css-loader',
            options: {
              modules: false
            }
          },
          // sass-loader
          {
            loader: 'sass-loader'
          }
        ]
      },
      // fonts loader
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/,
        use: [{
          loader: "file-loader",
          options: {
            name: "[name].[ext]",
            outputPath: 'font',
            publicPath: '../font'
          }
        }, ],
      },
      {
        test: /\.(png|jp(e*)g|svg)$/,
        use: [{
          loader: 'file-loader',
          options: {
            limit: 8000, // Convert images < 8kb to base64 strings
            name: '[hash]-[name].[ext]',
            outputPath: 'img/',
            publicPath: '../img/'
          }
        }]
      }
    ]
  },
  plugins: [
    new MiniCssExtractPlugin({
      filename: "css/style.css"
    }),
    // new CleanWebpackPlugin(),
  ],
  watch: true
};
